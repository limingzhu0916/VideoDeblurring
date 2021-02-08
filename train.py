import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.model import create_model
from util.metric_counter import MetricCounter
from multiprocessing import freeze_support
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train(opt, data_loader, model, visualizer):
    if opt.use_validation:
        dataset, validation = data_loader.load_data()
    else:
        dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('The number of training images = %d' % dataset_size)
    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        visualizer.clear()
        for i, data in enumerate(dataset):
            total_steps += opt.batchSize
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            losses = model.get_current_losses()
            visualizer.add_losses(losses['G_GAN'], losses['G_Content'], losses['D'])

            if total_steps % 100 == 0:
                curr_psnr, curr_ssim, img_for_vis = model.get_images_and_metrics()
                visualizer.add_image(img_for_vis, tag='train')  # display images on tensorboard
                visualizer.add_metrics(curr_psnr, curr_ssim)

        visualizer.write_to_tensorboard(epoch)

        if opt.use_validation and epoch % opt.validation_freq == 0:
            print('Testing.....')
            for i, data in enumerate(validation):
                val_psnr, val_ssim = model.test_validation(data)
                visualizer.add_metrics(val_psnr, val_ssim)
            visualizer.write_metrics_tensorboard(epoch, validation=True)

        if epoch % 400 == 0:
            model.save_networks('latest')
        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if epoch > opt.niter:
            model.update_learning_rate()


if __name__ == '__main__':
    freeze_support()
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    model = create_model(opt)
    visualizer = MetricCounter(opt.name)
    train(opt, data_loader, model, visualizer)
