/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package src.VipulSharma;

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
import org.opencv.highgui.HighGui;
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
public class SVMTrain_2 {

//    public static void main(String[] args) {
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//
//        String DATABASE = "yalefaces_aligned";
////        Net NET = Dnn.readNetFromTorch("openface.nn4.small2.v1.t7");
//
//        boolean CLAHE_ON = false;
//        boolean FACENET_ON = true;
//        boolean BIF_ON = false;
//        int BIF_bands = 8;
//        int BIF_rots = 8;
//
//        ArrayList<Integer> training_labels_array = new ArrayList<Integer>();
//        ArrayList<Integer> testing_labels_array = new ArrayList<Integer>();
//        Mat TRAINING_DATA = new Mat();
//        Mat TESTING_DATA = new Mat();
//
//        // Load training and testing data
//        File[] directories = new File(DATABASE).listFiles();
//        for (int i = 0; i < directories.length; i++) {
//            File[] files = directories[i].listFiles();
//            for (int j = 0; j < 5; j++) {
//                Mat image = Imgcodecs.imread(files[j].getAbsolutePath());
//                Mat training_feature = Feature_Extractor.extract_feature(image,
//                        CLAHE_ON, FACENET_ON, NET, BIF_ON, BIF_bands, BIF_rots);
//                TRAINING_DATA.push_back(training_feature);
//                training_labels_array.add((i + 1));
//            }
//            for (int j = 5; j < files.length; j++) {
//                Mat image = Imgcodecs.imread(files[j].getAbsolutePath());
//                Mat testing_feature = Feature_Extractor.extract_feature(image,
//                        CLAHE_ON, FACENET_ON, NET, BIF_ON, BIF_bands, BIF_rots);
//                TESTING_DATA.push_back(testing_feature);
//                testing_labels_array.add((i + 1));
//            }
//        }
//
//        // Put training and testing labels into Mats
//        Mat TRAINING_LABELS = Mat.zeros(TRAINING_DATA.rows(), 1, CvType.CV_8UC1);
//        for (int i = 0; i < training_labels_array.size(); i++) {
//            TRAINING_LABELS.put(i, 0, training_labels_array.get(i));
//        }
//        Mat TESTING_LABELS = Mat.zeros(TESTING_DATA.rows(), 1, CvType.CV_8UC1);
//        for (int i = 0; i < testing_labels_array.size(); i++) {
//            TESTING_LABELS.put(i, 0, testing_labels_array.get(i));
//        }
//
//        System.out.println("TRAINING_DATA - Rows:" + TRAINING_DATA.rows()
//                + " Cols:" + TRAINING_DATA.cols());
//        System.out.println("TRAINING_LABELS - Rows:" + TRAINING_LABELS.rows()
//                + " Cols:" + TRAINING_LABELS.cols());
//        //System.out.println(TRAINING_LABELS.dump());
//        System.out.println("TESTING_DATA - Rows:" + TESTING_DATA.rows()
//                + " Cols:" + TESTING_DATA.cols());
//        System.out.println("TESTING_LABELS - Rows:" + TESTING_LABELS.rows()
//                + " Cols:" + TESTING_LABELS.cols());
//        //System.out.println(TRAINING_LABELS.dump());
//
//        // Train SVM
//        SVM svm = SVM.create();
//        svm.setKernel(SVM.LINEAR);
//        svm.setType(SVM.C_SVC);
//        // errors here
//        svm.train(TRAINING_DATA, Ml.ROW_SAMPLE, TRAINING_LABELS);
//
//        Mat RESULTS = new Mat();
//        // do i need to predict test features one-by-one?
//        // what is flags?
//        svm.predict(TESTING_DATA, RESULTS, flags);
//    }
    //****************************************************************************//
//****************************************************************************//
//****************************************************************************//
//****************************************************************************//
//****************************************************************************//
//****************************************************************************//
//    public static void main(String[] args) {
//
////    }
////    public static void main(String[] args) {
//        // Load the native OpenCV library
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        // Set up training data
//        int[] labels = {1, -1, -1, -1};
//        float[] trainingData = {501, 10, 255, 10, 501, 255, 10, 501};
//        Mat trainingDataMat = new Mat(4, 2, CvType.CV_32FC1);
//        trainingDataMat.put(0, 0, trainingData);
//        Mat labelsMat = new Mat(4, 1, CvType.CV_32SC1);
//        labelsMat.put(0, 0, labels);
//        // Train the SVM
//        SVM svm = SVM.create();
//        svm.setType(SVM.C_SVC);
//        svm.setKernel(SVM.LINEAR);
//        svm.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER, 100, 1e-6));
//        svm.train(trainingDataMat, Ml.ROW_SAMPLE, labelsMat);
//        // Data for visual representation
//        int width = 512, height = 512;
//        Mat image = Mat.zeros(height, width, CvType.CV_8UC3);
//        // Show the decision regions given by the SVM
//        byte[] imageData = new byte[(int) (image.total() * image.channels())];
//        Mat sampleMat = new Mat(1, 2, CvType.CV_32F);
//        float[] sampleMatData = new float[(int) (sampleMat.total() * sampleMat.
//                channels())];
//        for (int i = 0; i < image.rows(); i++) {
//            for (int j = 0; j < image.cols(); j++) {
//                sampleMatData[0] = j;
//                sampleMatData[1] = i;
//                sampleMat.put(0, 0, sampleMatData);
//                float response = svm.predict(sampleMat);
//                if (response == 1) {
//                    imageData[(i * image.cols() + j) * image.channels()] = 0;
//                    imageData[(i * image.cols() + j) * image.channels() + 1]
//                            = (byte) 255;
//                    imageData[(i * image.cols() + j) * image.channels() + 2] = 0;
//                } else if (response == -1) {
//                    imageData[(i * image.cols() + j) * image.channels()]
//                            = (byte) 255;
//                    imageData[(i * image.cols() + j) * image.channels() + 1] = 0;
//                    imageData[(i * image.cols() + j) * image.channels() + 2] = 0;
//                }
//            }
//        }
//        image.put(0, 0, imageData);
//        // Show the training data
//        int thickness = -1;
//        int lineType = Core.LINE_8;
//        Imgproc.circle(image, new Point(501, 10), 5, new Scalar(0, 0, 0),
//                thickness, lineType, 0);
//        Imgproc.circle(image, new Point(255, 10), 5, new Scalar(255, 255, 255),
//                thickness, lineType, 0);
//        Imgproc.circle(image, new Point(501, 255), 5, new Scalar(255, 255, 255),
//                thickness, lineType, 0);
//        Imgproc.circle(image, new Point(10, 501), 5, new Scalar(255, 255, 255),
//                thickness, lineType, 0);
//        // Show support vectors
//        thickness = 2;
//        Mat sv = svm.getUncompressedSupportVectors();
//        float[] svData = new float[(int) (sv.total() * sv.channels())];
//        sv.get(0, 0, svData);
//        for (int i = 0; i < sv.rows(); ++i) {
//            Imgproc.circle(image, new Point(svData[i * sv.cols()], svData[i
//                    * sv.cols() + 1]), 6,
//                    new Scalar(128, 128, 128), thickness, lineType, 0);
//        }
//        Imgcodecs.imwrite("result.png", image); // save the image
//        HighGui.imshow("SVM Simple Example", image); // show it to the user
//        HighGui.waitKey();
//
//    }
}