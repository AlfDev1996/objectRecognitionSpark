package it.unisa.objectrecognitionspark.model;
import org.bytedeco.javacpp.*;
import org.bytedeco.opencv.global.opencv_dnn;

import org.bytedeco.opencv.opencv_dnn.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.javacpp.indexer.*;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;

import org.apache.commons.lang3.time.StopWatch;

import java.util.*;
import java.io.*;

public class ObjectRecognitor implements Serializable{

    List<String> list =new ArrayList<>();
    String[] Labels = null;
    byte[] byteCfg=null;
    byte[] byteModel=null;
    transient Net net=null; //Escludi dalla serializzazione poiché non necessario

    public ObjectRecognitor(byte[] byteCfg,byte[] byteModel,List<String> listNames){
        this.byteCfg=byteCfg;
        this.byteModel=byteModel;
        list = listNames;
        Labels = list.toArray(new String[list.size()]);
        net = opencv_dnn.readNetFromDarknet(byteCfg,byteModel);
        net.setPreferableBackend(3);

        net.setPreferableTarget(0);
    }

    public List<String> recognition(byte[] bytes){
        List<String> res=null;
        try{

            if(bytes==null) return null;

            if(net==null){
                net = opencv_dnn.readNetFromDarknet(this.byteCfg,this.byteModel);

                net.setPreferableBackend(3);

                net.setPreferableTarget(0);
            }
            Mat img = imdecode(new Mat(bytes), 1);

            Mat blob = opencv_dnn.blobFromImage(img, 1.0 / 255, new Size(416, 416), new Scalar(), true, false,CV_32F);

            //input data
            net.setInput(blob);

            StringVector outNames = net.getUnconnectedOutLayersNames();

            MatVector outs = new MatVector();
            for(int i=0;i<outNames.size();i++){
                outs.put(new Mat());
            }

            //forward model
            StopWatch sw = StopWatch.createStarted();
            net.forward(outs, outNames);
            sw.stop();

            //get result from all output
            float threshold = 0.5f;       //for confidence
            float nmsThreshold = 0.3f;    //threshold for nms
            res = GetResult(outs, img, threshold, nmsThreshold, true);
            System.gc();
            return res;
        }catch (Exception ex){
            ex.printStackTrace();
        }finally {
            return res;
        }
    }

    private List<String> GetResult(MatVector output, Mat image, float threshold, float nmsThreshold, boolean nms)
    {
        List<String> outputList=null;
        nms = true;
        //for nms
        ArrayList<Integer> classIds = new ArrayList<>();
        ArrayList<Float> confidences = new ArrayList<>();
        ArrayList<Float> probabilities = new ArrayList<>();
        ArrayList<Rect2d> rect2ds = new ArrayList<>();
        //Rect2dVector boxes = new Rect2dVector();
        try{
            int w = image.cols();
            int h = image.rows();
            /*
             YOLO3 COCO trainval output
             0 1 : center                    2 3 : w/h
             4 : confidence                  5 ~ 84 : class probability
            */
            int prefix = 5;   //skip 0~4
            for(int k=0;k<output.size();k++)
            {
                Mat prob = output.get(k);
                final FloatRawIndexer probIdx = prob.createIndexer();
                for (int i = 0; i < probIdx.rows(); i++)
                {
                    float confidence = probIdx.get(i, 4);
                    if (confidence > threshold)
                    {
                        //get classes probability
                        DoublePointer minVal= new DoublePointer();
                        DoublePointer maxVal= new DoublePointer();
                        Point min = new Point();
                        Point max = new Point();
                        minMaxLoc(prob.rows(i).colRange(prefix, prob.cols()), minVal, maxVal, min, max, null);
                        int classes = max.x();
                        float probability = probIdx.get(i, classes + prefix);

                        if (probability > threshold) //more accuracy, you can cancel it
                        {
                            //get center and width/height
                            float centerX = probIdx.get(i, 0) * w;
                            float centerY = probIdx.get(i, 1) * h;
                            float width = probIdx.get(i, 2) * w;
                            float height = probIdx.get(i, 3) * h;
                            if (!nms)
                            {
                                continue;
                            }

                            //put data to list for NMSBoxes
                            classIds.add(classes);
                            confidences.add(confidence);
                            probabilities.add(probability);
                            rect2ds.add(new Rect2d(centerX, centerY, width, height));
                        }
                    }
                }
            }

            if (!nms) return outputList;

            //using non-maximum suppression to reduce overlapping low confidence box
            IntPointer indices = new IntPointer(confidences.size());
            Rect2dVector boxes = new Rect2dVector();
            for(int i=0;i<rect2ds.size();i++){
                boxes.push_back(rect2ds.get(i));
            }

            FloatPointer con = new FloatPointer(confidences.size());
            float[] cons = new float[confidences.size()];
            for(int i=0;i<confidences.size();i++){
                cons[i] = confidences.get(i);
            }
            con.put(cons);
            NMSBoxes(boxes, con, 0.5f, 0.4f, indices);

            outputList=new ArrayList<>();
            for (int m=0;m<indices.limit();m++)
            {
                int i = indices.get(m);
                outputList.add(Labels[classIds.get(i)]);
            }
            return outputList;
        }catch(Exception e){
            e.printStackTrace();
        }finally {
            return outputList;
        }
    }
}
