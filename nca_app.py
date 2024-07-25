# TODO
# Create enums for app and model modes
# Centralise hyperparameters in config
# Improve recording file metadata
# Separate UI application loop and NCA eval with multiprocessing

import sys, os, random, time

from modular_nca import config, viewer, hex_renderer, hex_neural_model, hex_dataset, hex_trainer

APP_MODE = int(sys.argv[1])
# -1 = train
# 0 = preview dataset
# 1 = eval rgb
# 2 = eval alpha
# 3 = eval hidden layers 1-3
# 4 = eval hidden layers 4-6


MODEL_MODE = 0
# 0 = time series
# 1 = reconstruction
# 2 = batch

SEED_SELECTION = None

reload_progress = False
if APP_MODE == -1:
    reload_progress = input("Reload progress? (y/n)") == "y"

if __name__ == "__main__":

    model = hex_neural_model.HexNeuralModel(num_hidden_layers=12, nn_hidden_layer_sizes=(96, ), divisions=110)
    renderer = hex_renderer.HexRenderer((600, 600), model)
    dataset = hex_dataset.HexDataset(model)
    view = viewer.Viewer(renderer)

    if MODEL_MODE == 0:
        dataset.load_time_series(config.dataset_path, start=0, end=1, stride=1, blur_strength=0.012)
    
    elif MODEL_MODE == 1:
        dataset.load_reconstruction_set(config.dataset_path, dataset_coverage=1, blur_strength=0.01)

    if APP_MODE < 1:

        if MODEL_MODE == 2:
            dataset.load_and_sample_images(config.dataset_path, sample_coverage=0.4, samples_per_image=2000, blur_strength=0.01)
        
        model.reset_grid_seed(dataset.samples[-1])
        #model.preview_alpha(dataset.samples[-1])

    if APP_MODE == -1:

        if reload_progress:
            eval_epoch_num = sorted([int(f.split("-")[-1].split(".")[0]) for f in os.listdir(config.checkpoint_dir)])[-1]

            print("Loading model epoch %d" %eval_epoch_num)
            model.load_nn(eval_epoch_num)

        trainer = hex_trainer.HexNNTrainer(model, dataset)
        #trainer.learn_batch(num_epochs=500)

        if MODEL_MODE == 0:
            trainer.learn_time_series(num_epochs=10000, stability_pre_iters=0, stability_post_iters=4)

        elif MODEL_MODE == 1:
            trainer.learn_reconstruction_goal(num_epochs=10000, stability_pre_iters=0, stability_post_iters=0)

    if APP_MODE > 0:

        eval_epoch_num = sorted([int(f.split("-")[-1].split(".")[0]) for f in os.listdir(config.checkpoint_dir)])[-1]

        print("Loading model epoch %d" %eval_epoch_num)
        model.load_nn(eval_epoch_num)

        if MODEL_MODE == 0:
            model.set_mask(dataset.masks[0])
            model.reset_grid_seed(dataset.samples[0])

        if MODEL_MODE == 1:
            
            seed_index = SEED_SELECTION
            if SEED_SELECTION is None:
                seed_index = random.randint(0, int(len(dataset.samples)/2)-1) * 2
            
            print("Picked seed no. %d" %(seed_index/2))

            model.set_mask(dataset.masks[int(seed_index/2)])
            model.reset_grid_seed(dataset.samples[seed_index])

        elif MODEL_MODE == 2:
            model.reset_grid_rand()

        info = ((config.dataset_path.split("/")[-2]), APP_MODE, MODEL_MODE, model.nn.hidden_layer_sizes[0], model.num_layers, model.cell_count, eval_epoch_num, time.time())
        view.set_recording_settings(name=config.recording_dir+"hex-%s-%d-%d-%d-%d-%d-%d-%d" %info, frame_count=500, fps=24)
    
    if APP_MODE != -1:
        print("\nLaunching viewer...\nLeft click to select a cell.\nRight click to toggle dragging.\nScroll to zoom.\nPress 'q' to quit.\n")
        on_frame_func = model.step if APP_MODE > 0 else None
        view.start(mode=APP_MODE, on_frame=on_frame_func)
