import copy
import math
import os
from functools import partial
import wandb
import torch
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from datasets.loader import construct_loader
from utils.parsing import parse_train_args
from utils.training import train_epoch, test_epoch, loss_function, inference_epoch_fix, cfr_loss_function
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model, ExponentialMovingAverage


def train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir):
    best_val_loss = math.inf
    best_val_inference_value = math.inf if args.inference_earlystop_goal == 'min' else 0
    best_epoch = 0
    best_val_inference_epoch = 0
    # 根据 confidence_mode 选择损失函数
    loss_fn = partial(cfr_loss_function) if args.confidence_mode else partial(
        loss_function, tr_weight=args.tr_weight, rot_weight=args.rot_weight,
        tor_weight=args.tor_weight, no_torsion=args.no_torsion
    )

    train_loss_history = []
    val_loss_history = []
    epochs = []

    print("Starting training...")
    print("Run directory:", run_dir)
    print("Current working directory:", os.getcwd())

    for epoch in range(args.n_epochs):
        if epoch % 5 == 0:
            print("Run name: ", args.run_name)
        logs = {}
        train_losses = train_epoch(
            model, train_loader, optimizer, args.device, t_to_sigma, loss_fn, ema_weights,
            confidence_mode=args.confidence_mode
        )
        print("Epoch {}: Training loss {:.4f}{}".format(
            epoch, train_losses['loss'],
            "" if args.confidence_mode else f"  tr {train_losses['tr_loss']:.4f}   rot {train_losses['rot_loss']:.4f}   tor {train_losses['tor_loss']:.4f}"
        ))

        ema_weights.store(model.parameters())
        if args.use_ema:
            ema_weights.copy_to(model.parameters())
        val_losses = test_epoch(
            model, val_loader, args.device, t_to_sigma, loss_fn, args.test_sigma_intervals,
            confidence_mode=args.confidence_mode
        )
        print("Epoch {}: Validation loss {:.4f}{}".format(
            epoch, val_losses['loss'],
            "" if args.confidence_mode else f"  tr {val_losses['tr_loss']:.4f}   rot {val_losses['rot_loss']:.4f}   tor {val_losses['tor_loss']:.4f}"
        ))

        train_loss_history.append(train_losses['loss'])
        val_loss_history.append(val_losses['loss'])
        epochs.append(epoch)

        if args.val_inference_freq is not None and (epoch + 1) % args.val_inference_freq == 0:
            inf_metrics = inference_epoch_fix(
                model, val_loader.dataset.complex_graphs[:args.num_inference_complexes],
                args.device, t_to_sigma, args
            )
            print("Epoch {}: Val inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f}".format(
                epoch, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5']
            ))
            logs.update({'valinf_' + k: v for k, v in inf_metrics.items()}, step=epoch + 1)

        if not args.use_ema:
            ema_weights.copy_to(model.parameters())
        ema_state_dict = copy.deepcopy(model.module.state_dict() if args.device.type == 'cuda' else model.state_dict())
        ema_weights.restore(model.parameters())

        if args.wandb:
            logs.update({'train_' + k: v for k, v in train_losses.items()})
            logs.update({'val_' + k: v for k, v in val_losses.items()})
            logs['current_lr'] = optimizer.param_groups[0]['lr']
            wandb.log(logs, step=epoch + 1)

        state_dict = model.module.state_dict() if args.device.type == 'cuda' else model.state_dict()
        if args.inference_earlystop_metric in logs.keys() and \
                (args.inference_earlystop_goal == 'min' and logs[args.inference_earlystop_metric] <= best_val_inference_value or
                 args.inference_earlystop_goal == 'max' and logs[args.inference_earlystop_metric] >= best_val_inference_value):
            best_val_inference_value = logs[args.inference_earlystop_metric]
            best_val_inference_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_inference_epoch_model.pt'))
            torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_inference_epoch_model.pt'))
        if val_losses['loss'] <= best_val_loss:
            best_val_loss = val_losses['loss']
            best_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_model.pt'))
            torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_model.pt'))

        if scheduler:
            if args.val_inference_freq is not None:
                scheduler.step(best_val_inference_value)
            else:
                scheduler.step(val_losses['loss'])

        torch.save({
            'epoch': epoch,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'ema_weights': ema_weights.state_dict(),
        }, os.path.join(run_dir, 'last_model.pt'))

        if (epoch + 1) % 2 == 0:
            print(f"Generating loss curve plot at epoch {epoch}")
            plot_loss_curve(epochs, train_loss_history, val_loss_history, run_dir)

    print("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
    print("Best inference metric {} on Epoch {}".format(best_val_inference_value, best_val_inference_epoch))

def plot_loss_curve(epochs, train_loss, val_loss, run_dir):
    print("Plotting loss curve...")
    print("Current working directory:", os.getcwd())
    print("Run directory:", run_dir)

    filtered_epochs = []
    filtered_train_loss = []
    filtered_val_loss = []
    for e, t_loss, v_loss in zip(epochs, train_loss, val_loss):
        if not (math.isfinite(t_loss) and math.isfinite(v_loss)):
            print(f"Skipping epoch {e} because of invalid loss: train_loss={t_loss}, val_loss={v_loss}")
            continue
        diff = v_loss - t_loss
        if diff <= 100:  # 放宽阈值到100
            filtered_epochs.append(e)
            filtered_train_loss.append(t_loss)
            filtered_val_loss.append(v_loss)
        else:
            print(f"Skipping epoch {e} because val_loss - train_loss = {diff:.4f} > 100, val_loss={v_loss:.4f}, train_loss={t_loss:.4f}")

    if not filtered_epochs:
        print("No data points to plot after filtering (val_loss - train_loss <= 100).")
        return

    plt.figure(figsize=(10, 6))
    print("Figure created.")
    plt.plot(filtered_epochs, filtered_train_loss, label='Train Loss', color='blue', marker='o')
    plt.plot(filtered_epochs, filtered_val_loss, label='Validation Loss', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs (Filtered: val_loss - train_loss <= 100)')
    plt.legend()

    for i, txt in enumerate(filtered_train_loss):
        plt.annotate(f'{txt:.2f}', (filtered_epochs[i], filtered_train_loss[i]), textcoords="offset points",
                     xytext=(0, 10), ha='center', color='blue')
    for i, txt in enumerate(filtered_val_loss):
        plt.annotate(f'{txt:.2f}', (filtered_epochs[i], filtered_val_loss[i]), textcoords="offset points",
                     xytext=(0, -15), ha='center', color='red')

    result_dir = os.path.join(run_dir, 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print("Created result directory:", result_dir)
    else:
        print("Result directory already exists:", result_dir)

    save_path = os.path.join(result_dir, 'loss_curve.png')
    print("Saving plot to:", save_path)
    plt.savefig(save_path)
    print("Plot saved successfully to:", save_path)
    plt.close()

def main_function():
    args = parse_train_args()
    print("args.log_dir:", args.log_dir)
    print("args.run_name:", args.run_name)

    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        args.config = args.config.name

    assert (args.inference_earlystop_goal == 'max' or args.inference_earlystop_goal == 'min')
    if args.val_inference_freq is not None and args.scheduler is not None:
        assert (args.scheduler_patience > args.val_inference_freq)
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    args.n_epochs = 20

    # 直接使用 t_to_sigma_compl
    t_to_sigma = t_to_sigma_compl

    # 传递 sigma 参数给 construct_loader
    train_loader, val_loader, val_dataset2 = construct_loader(
        args,
        t_to_sigma,
        device,
        tr_sigma_min=getattr(args, 'tr_sigma_min', 0.1),
        tr_sigma_max=getattr(args, 'tr_sigma_max', 10.0),
        rot_sigma_min=getattr(args, 'rot_sigma_min', 0.03),
        rot_sigma_max=getattr(args, 'rot_sigma_max', 1.55),
        tor_sigma_min=getattr(args, 'tor_sigma_min', 0.03),
        tor_sigma_max=getattr(args, 'tor_sigma_max', 3.14)
    )

    model = get_model(args, device, t_to_sigma=t_to_sigma)
    optimizer, scheduler = get_optimizer_and_scheduler(
        args, model,
        scheduler_mode=args.inference_earlystop_goal if args.val_inference_freq is not None else 'min'
    )
    ema_weights = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)

    if args.restart_dir:
        try:
            dict = torch.load(f'{args.restart_dir}/last_model.pt', map_location=torch.device('cpu'))
            if args.restart_lr is not None:
                dict['optimizer']['param_groups'][0]['lr'] = args.restart_lr
            optimizer.load_state_dict(dict['optimizer'])
            model.module.load_state_dict(dict['model'], strict=True)
            if hasattr(args, 'ema_rate'):
                ema_weights.load_state_dict(dict['ema_weights'], device=device)
            print("Restarting from epoch", dict['epoch'])
        except Exception as e:
            print("Exception", e)
            dict = torch.load(f'{args.restart_dir}/best_model.pt', map_location=torch.device('cpu'))
            model.module.load_state_dict(dict, strict=True)
            print("Due to exception had to take the best epoch and no optimiser")

    numel = sum([p.numel() for p in model.parameters()])
    print('Model with', numel, 'parameters')

    if args.wandb:
        wandb.init(
            entity='entity',
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=args
        )
        wandb.log({'numel': numel})

    run_dir = os.path.join(args.log_dir, args.run_name)
    print("Run directory initialized as:", run_dir)
    yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, args.__dict__)
    args.device = device

    train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main_function()