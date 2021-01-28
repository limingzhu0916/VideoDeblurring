from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.model import create_model
import time
from util import util

def test(opt, data_loader, model):
    dataset = data_loader.load_data()
    test_time = 0.0
    iteration = 0.0
    avgPSNR = 0.0
    avgSSIM = 0.0
    counter = 0
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        start = time.time()
        model.test()
        stop = time.time()
        test_time += stop - start
        print('RunTime:%.4f' % (stop - start), '  Average Runtime:%.4f' % (test_time / (iteration + 1)))
        fake, psnr, ssim = model.get_images_and_metrics()
        util.save_image(fake, opt.results_dir)
        avgPSNR += psnr
        avgSSIM += ssim
        counter = i + 1
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
    avgPSNR /= counter
    avgSSIM /= counter
    print('PSNR = %f, SSIM = %f' % (avgPSNR, avgSSIM))


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle

    data_loader = CreateDataLoader(opt)
    model = create_model(opt)
    test(opt, data_loader, model)