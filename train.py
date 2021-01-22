import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.model import create_model
from util.metric_counter import MetricCounter
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)

    visualizer = MetricCounter(opt.name)

    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        visualizer.clear()
        for i, data in enumerate(dataset):
            total_steps += opt.batchsize
            epoch_iter += opt.batchsize
            model.set_input(data)    # unpack data from dataset and apply preprocessing
            model.optimize_parameters()     # calculate loss functions, get gradients, update network weights

            if total_steps % opt.display_freq == 0:  # display images on tensorboard
                save_result = total_steps % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_steps % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_steps))
                save_suffix = 'iter_%d' % total_steps if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))




