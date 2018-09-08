/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package VipulSharma;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.ResourceBundle;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.dnn.Dnn;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import org.opencv.ml.SVMSGD;
import org.opencv.ml.TrainData;
import org.opencv.videoio.VideoCapture;

/**
 *
 * @author Andika Mulyawan
 */
public class SVMTrain {

    public static final String SVM_MODEL_FILE_PATH
            = "E:\\TA";

    public static final String SVM_ROOT = "";

    public static final String SVM_HAS_TEST = "E:\\TA\\Hand";
//    public static final String SVM_HAS_TEST = "has/test";
    public static final String SVM_HAS_TRAIN = "E:\\TA\\Hand";

    public static final String SVM_NO_TEST = "E:\\TA\\noHand";
//    public static final String SVM_NO_TEST = "no/test";
    public static final String SVM_NO_TRAIN = "E:\\TA\\noHand";

//    public static void main(String[] args) {
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        SVMTrain vl = new SVMTrain();
//        vl.train();
//
//    }
    void train() {

        try {
            SVM svm = SVM.create();
            TrainData t;

            Mat labelsMat = new Mat(4, 1, CvType.CV_32SC1);
//            labelsMat.put(0, 0, labels);//        svm.setType(SVM.C_SVC);
            svm.setKernel(SVM.POLY);
//        svm.setDegree(0.1);
//        // 1.4 bug fix: old 1.4 ver gamma is 1
//        svm.setGamma(0.1);
//        svm.setCoef0(0.1);
//        svm.setC(1);
//        svm.setNu(0.1);
//        svm.setP(0.1);
            svm.setTermCriteria(new TermCriteria(1, 5, 0.0001));

//            TrainData trainData = loadTrainData();
            Mat trainData2 = PreImage();
//            TrainData trainData = TrainData.create(trainData2, 0, trainData2)
            long star = System.currentTimeMillis();
            System.out.println("start train...");

            Mat labels = new Mat(hasFileList.size(), 1, CvType.CV_32FC1);
            System.out.println("end train...total time ： " + (System.
                    currentTimeMillis() - star) + "ms");

//        svm.save(SVM_MODEL_FILE_PATH);
            System.out.println("save the train model...");
        } catch (Exception e) {
            System.out.println("train()");
            System.out.println(e);
            System.out.println("");
        }

    }

    /**
     *
     *
     * @return
     */
    TrainData loadTrainData() {
        Mat trainingData = new Mat();
        Mat classes = new Mat();
        try {
            List<File> hasFileList = getFiles(SVM_ROOT + SVM_HAS_TRAIN);
            List<File> noFileList = getFiles(SVM_ROOT + SVM_NO_TRAIN);
            int hasCount = hasFileList.size();
            int noCount = noFileList.size();
            Mat samples = new Mat();
            Mat labels = new Mat();

            for (int i = 0; i < hasCount; i++) {//positive
                System.out.println(hasFileList.get(i).getAbsolutePath());
                Mat img = getMat(hasFileList.get(i).getAbsolutePath());
                samples.push_back(img.reshape(1, 1));
                labels.push_back(Mat.ones(new Size(1, 1), CvType.CV_32F));
            }

            for (int j = 0; j < noCount; j++) {//negative
                System.out.println(noFileList.get(j).getAbsolutePath());
                Mat img = getMat(noFileList.get(j).getAbsolutePath());
                samples.push_back(img.reshape(1, 1));
                labels.push_back(Mat.zeros(new Size(1, 1), CvType.CV_32F));
            }

            samples.copyTo(trainingData);
//
//            trainingData.convertTo(trainingData, CvType.CV_32F);
//
            labels.copyTo(classes);
        } catch (Exception e) {
            System.out.println("loadTrainData()");
            System.out.println(e);
            System.out.println("");
        }

        return TrainData.create(trainingData, Ml.ROW_SAMPLE, classes);
    }
    public List<File> hasFileList;
    public List<Mat> preTrain;

    public Mat PreImage() {
        hasFileList = getFiles(SVM_ROOT + SVM_HAS_TRAIN);
        preTrain = new ArrayList<>();
        for (int i = 0; i < hasFileList.size(); i++) {//positive
            Mat img = getMat(hasFileList.get(i).getAbsolutePath());
            Core.flip(img, img, 1);
            img = getBox(img);
            preTrain.add(img);
        }

        Mat trainingData = new Mat(hasFileList.size(), preTrain.get(0).rows()
                * preTrain.get(0).cols(), CvType.CV_32FC1);
        int ii = 0; // Current column in training_mat
        for (File file : hasFileList) {
            Mat img_mat = getMat(file.getAbsolutePath());
            for (int i = 0; i < preTrain.get(0).rows(); i++) {
                for (int j = 0; j < preTrain.get(0).cols(); j++) {
//                    training_mat.at<float>(i ,ii++) = img_mat.at<uchar>(i,j
//                    );
                    trainingData.put(i, ii, img_mat.get(i, j));
                    ii++;
                }
            }
        }
        return trainingData;
    }

    private Mat getBox(Mat frame) {
        Rect rectCrop = new Rect(new Point(frame.cols() - 5, 10 + 5), new Point(
                frame.cols() / 2 + 5, frame.rows() - (frame.rows() / 3) - 10 - 5)
        );
        frame = frame.submat(rectCrop);
        //# convert the roi to grayscale and blur it
//        Imgproc.accumulateWeighted(frame, frame, 0.5);
        Imgproc.GaussianBlur(frame, frame, new Size(7, 7), 0);
        return frame;
    }
//////

    public Mat getMat(String path) {
        Mat img = new Mat();
        Mat con = Imgcodecs.imread(path);
        try {

            con = Imgcodecs.imread(path, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
            con.convertTo(img, CvType.CV_32F, 1.0 / 255.0);
        } catch (Exception e) {
            System.out.println("getMat(String path)");
            System.out.println(e);
            System.out.println("");
        }
        return con;
    }

//    public Mat getHistogramFeatures(Mat image) {
//        Mat grayImage = new Mat();
//        Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_RGB2GRAY);
//
//        Mat img_threshold = new Mat();
//        Imgproc.threshold(grayImage, img_threshold, 0, 255,
//                Imgproc.THRESH_OTSU
//                + Imgproc.THRESH_BINARY);
//        return img_threshold;
//    }
    /**
     * @param floderPath
     * @return
     */
    public List<File> getFiles(String floderPath) {
        List<File> list = new ArrayList<File>();

        File file = new File(floderPath);

        if (!file.exists()) {
            System.out.
                    println("Error : " + floderPath + " folder is not exist!");
            return list;
        }

        if (!file.isDirectory()) {
            System.out.println("Error : " + floderPath + "  is not a folder!");
            return list;
        }

        File[] files = file.listFiles();
        if (files.length == 0) {
            System.out.println("Error : " + floderPath + "  folder is empty!");
            return list;
        }

        for (int i = 0; i < files.length; i++) {
            File f = files[i];
            list.add(f);
        }
        return list;
    }

}
