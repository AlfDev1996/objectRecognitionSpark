#! /bin/sh

spark-submit --class it.unisa.objectrecognitionspark.logic.RunProjectSequenceFile \
  --master yarn --deploy-mode cluster --driver-memory 4g \
  --num-executors 8 --executor-memory 13g --executor-cores 4 \
  ObjectRecognitionSpark-1.0-SNAPSHOT-jar-with-dependencies.jar \
  /dragoneriannaproject/Images/ \
  /dragoneriannaproject/test_exec \
  2
