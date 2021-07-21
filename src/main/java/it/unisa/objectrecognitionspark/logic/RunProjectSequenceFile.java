package it.unisa.objectrecognitionspark.logic;

import it.unisa.objectrecognitionspark.model.ObjectRecognitor;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.io.BytesWritable;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

//Main class
public class RunProjectSequenceFile {
    public static void main(String[] args) throws Exception {
        //Lettura dalle risorse dei file di configurazione per la DNN OpenCv YoLoV3
        ClassLoader classLoader = RunProjectSequenceFile.class.getClassLoader();
        InputStream in = classLoader.getResourceAsStream("yolov3.cfg");
        final byte[] byteYoloCfg = IOUtils.toByteArray(in);

        in = classLoader.getResourceAsStream("yolov3.weights");
        final byte[] byteYoloWeights = IOUtils.toByteArray(in);

        final List<String> list=new ArrayList<>();
        in = classLoader.getResourceAsStream("coco.names");
        BufferedReader reader = new BufferedReader(new InputStreamReader(in));
        String line = reader.readLine();
        while(line!=null){
            list.add(line);
            line = reader.readLine();
        }


        String pathImg = args[0];
        String pathOutput = args[1];

        //Inizializzazione spark context
        SparkConf sparkConf = new SparkConf().setAppName("ObjectRecognitionSpark").set("spark.hadoop.validateOutputSpecs", "true").setMaster("local[0]");
        JavaSparkContext ctx = new JavaSparkContext(sparkConf);

        //Send in broadcast della configurazione necessaria per inizializzare le DNN OpenCv verso gli altri worker
        final Broadcast<byte[]> broadcastCfg = ctx.broadcast(byteYoloCfg);
        final Broadcast<byte[]> broadcastWeights = ctx.broadcast(byteYoloWeights);
        final Broadcast<List> broadcastList = ctx.broadcast(list);

        //Inizializzazione del modello
        ObjectRecognitor objectRecognitor=new ObjectRecognitor(broadcastCfg.value(),broadcastWeights.value(),broadcastList.value());

        //Primo Job, Lettura da sequence file
        JavaPairRDD<String, BytesWritable> images = ctx.sequenceFile(pathImg, String.class, BytesWritable.class);


        //Secondo Job, Map del contenuto letto in Byte[]
        JavaRDD<byte[]> byt = images.map(stringTuple2 -> stringTuple2._2().getBytes());


        //Terzo Job, Riconoscimento degli oggetti. 1 Task per Immagine
        JavaRDD<String> str = byt.flatMap((FlatMapFunction<byte[], String>) value -> {
            List<String> resultRecognition = objectRecognitor.recognition(value);
            return resultRecognition!=null ? resultRecognition.iterator() : null;
        });

        //Eventuali query che è possibile effettuare per filtrare le immagini su uno specifico oggetto riconosciuto
        //JavaRDD<String> str = str_tmp.filter(s -> s.equals("person"));

        //Quarto Job, trasformazione un coppia Stringa,Intero degli oggetti riconosciuti
        JavaPairRDD<String, Integer> ones = str.mapToPair(s -> new Tuple2<>(s, 1));

        //Quinto Job, Conteggio tramite riduzione degli oggetti riconosciuti
        JavaPairRDD<String, Integer> counts = ones.reduceByKey((i1, i2) -> i1 + i2);

        //Possibilità di eleggere l'oggetto riconosciuto + volte
        //Tuple2<String, Integer> maxV = counts.max((o1, o2) -> o1._2() > o2._2() ? o1._2() : o2._2());



        //Salvataggio in un solo file del risultato.
        counts.coalesce(1).saveAsTextFile(pathOutput);

        ctx.stop();

    }
}
