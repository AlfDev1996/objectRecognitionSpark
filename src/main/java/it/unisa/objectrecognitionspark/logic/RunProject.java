package it.unisa.objectrecognitionspark.logic;

import it.unisa.objectrecognitionspark.model.ObjectRecognitor;
import org.apache.commons.io.IOUtils;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.sql.*;
import scala.Tuple2;

import java.io.*;
import java.util.ArrayList;
import java.util.List;



//Classe Obsoleta
public class RunProject {
    public static void main(String[] args) throws Exception {
        ClassLoader classLoader = RunProject.class.getClassLoader();
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
        String sminP=args[2];
        int minP = Integer.parseInt(sminP);

        SparkSession spark = SparkSession
                .builder()
                .appName("ObjectRecognitionSpark")
                .getOrCreate();

        final Broadcast<byte[]> broadcastCfg = spark.sparkContext().broadcast(byteYoloCfg,akka.japi.Util.classTag(byte[].class));

        final Broadcast<byte[]> broadcastWeights = spark.sparkContext().broadcast(byteYoloWeights,akka.japi.Util.classTag(byte[].class));

        final Broadcast<List> broadcastList = spark.sparkContext().broadcast(list, akka.japi.Util.classTag(List.class));



        JavaRDD<Tuple2<String, PortableDataStream>> ll = spark.sparkContext().binaryFiles(pathImg,minP).toJavaRDD();

        JavaRDD<byte[]> byt = ll.map(stringPortableDataStreamTuple2 -> stringPortableDataStreamTuple2._2.toArray());

		ObjectRecognitor objectRecognitor=new ObjectRecognitor(broadcastCfg.value(),broadcastWeights.value(),broadcastList.value());
		
        JavaRDD<String> str = byt.flatMap((FlatMapFunction<byte[], String>) value -> objectRecognitor.recognition(value).iterator());


        JavaPairRDD<String, Integer> ones = str.mapToPair(s -> new Tuple2<>(s, 1));

        JavaPairRDD<String, Integer> counts = ones.reduceByKey((i1, i2) -> i1 + i2);

        counts.coalesce(1).saveAsTextFile(pathOutput);

        spark.stop();

    }
}
