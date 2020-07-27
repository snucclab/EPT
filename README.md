# Point to the Expression: 

- Preprocess datasets

```bash
python run_preprocess.py -I <INPUTDIR> -O <OUTPUTDIR> --alg <ALGFILE> --draw <DRAWFILE> --mawps <MAWPSFILE>
```

- Test measurement
```bash
python measure_test.py <DATASET FILES> ... --log <LOGDIR>
```

- Run experiment
```bash
./run_batch_tune.sh [base|large|xlarge] 
```