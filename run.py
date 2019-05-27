import src
import time
import warnings
warnings.filterwarnings("ignore")

start = time.time()
print('Running preprocessing...')
src.prepare_data.main('data')
print(f'Time elapsed {(time.time() - start)/60:.2f} minutes')

print('Training model and making submission...')
src.train_catboost.main('data')
print(f'Time elapsed {(time.time() - start)/60:.2f} minutes')

print('The end of execution')

