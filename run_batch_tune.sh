SIZE=$1

# Run MAWPS
python tune_hyperopt.py -s ./dataset/mawps_fold0 -n 1 -c ./config/${SIZE}.json -T 100 \
  -o 'num_decoder_layers=[3,6,9]' 'lr=[(2 ** (x/2)) / 800 for x in range(3)]' 'epoch_warmup=[100 * i / 100 for i in range(6)]' 'batch=1024' 'gradient_accumulation_steps=[1,2,4,8]'
python batch_trainer.py --set ./dataset/mawps --config ./config/mawps/base.json --model ept expr vanilla

# Run ALG514
python tune_hyperopt.py -s ./dataset/alg514_fold0 -n 1 -c ./config/${SIZE}.json -T 500 \
  -o 'num_decoder_layers=[3,6,9]' 'lr=[(2 ** (x/2)) / 800 for x in range(3)]' 'epoch_warmup=[500 * i / 100 for i in range(6)]' 'batch=1024' 'gradient_accumulation_steps=[1,2,4,8]'
python batch_trainer.py --set ./dataset/alg514 --config ./config/alg514/base.json --model ept expr vanilla

# Run DRAW
python tune_hyperopt.py -s ./dataset/draw -n 1 -c ./config/${SIZE}.json -T 500 \
  -o 'num_decoder_layers=[3,6,9]' 'lr=[(2 ** (x/2)) / 800 for x in range(3)]' 'epoch_warmup=[500 * i / 100 for i in range(6)]' 'batch=1024' 'gradient_accumulation_steps=[1,2,4,8]'
python batch_trainer.py --set ./dataset/draw --config ./config/draw/base.json --model ept expr vanilla
