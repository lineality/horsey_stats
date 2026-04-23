# horsey_stats

In 2026 the world was stunned \
by the steam-release of 'horsey game' \
an immersive VR-world-simulator the likes of which \
the multiverse had never dared to dream; \
global macro-economics and geo-politics \
would never be the same.

Here is a (very preliminary) tree-model \
implemented in vanilla rust \
to examine the otherworldly wizardy \
of: horsey game

# Command Line

Steps:
```bash
cargo test
cargo run -- train
cargo run -- predict
```

# Using Files: .train & .predict

### Add to training/testing/validation data:

As games are run, record results in path:

```path
/horsey_stats/data/test_train_data.csv
```
E.g.
```csv
row_id,game_id,age,height,experience,weight,rank,completion
0,1,7,143,2,727,1,1
1,1,3,150,2,856,3,1
2,1,8,143,4,765,4,1
3,1,4,163,2,904,2,1
4,2,3,165,1,967,0,0
5,2,6,166,1,1156,1,1
6,2,5,144,1,840,2,1
7,2,2,151,1,956,0,0
```

### Provide starting values for a prediction:

To predict the outcome before a game ends:

```path
/horsey_stats/data/predict.csv
```

E.g.
```csv
row_id,game_id,age,height,experience,weight,rank,completion
64,17,2,134,1,642,,
65,17,2,115,0,374,,
66,17,2,159,1,1419,,
67,17,6,128,2,643,,
```
