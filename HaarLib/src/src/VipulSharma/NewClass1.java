/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package src.VipulSharma;

import java.io.File;
import java.time.LocalTime;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.objdetect.HOGDescriptor;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;
import src.utils.Preprocessing;

/**
 *
 * @author Andika Mulyawan
 */
public class NewClass1 {

    /*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
     */
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        LocalTime time = java.time.LocalTime.now();
        System.out.println("time : " + time);
        File folder = new File("E:\\TA\\HandLearnSVM\\Try1\\Try1HandFull");
        HOGDescriptor gDescriptor = new HOGDescriptor();
        MatOfFloat svm = new MatOfFloat();
//        gDescriptor.setSVMDetector(svm);
//        Mat image = Imgcodecs.imread("C:\\Users\\Andika Mulyawan\\Desktop\\1.jpg");
        File[] listOfFiles = folder.listFiles();
        Mat trainingDataMat;

        trainingDataMat = new Mat(listOfFiles.length, 192780, CvType.CV_32FC1);

//        for (int i = 0; i < listOfFiles.length; i++) {
        Mat hand = Imgcodecs.imread(folder.getPath() + "\\" + listOfFiles[1].getName(), CvType.CV_32F);

        Imgproc.resize(hand, hand, new Size(192, 144));
        MatOfFloat descriptors = new MatOfFloat();
        gDescriptor.compute(hand, descriptors);
        for (int i = 0; i < descriptors.rows(); i++) {
            System.out.println(descriptors.get(i, 0)[0]);
        }
        System.out.println("PAPAP");
        float[] trainingData = descriptors.toArray();
        for (float f : trainingData) {
            System.out.println(f);
        }
//        trainingDataMat.put(i, 0, trainingData);
//        }
        LocalTime timeEnd = java.time.LocalTime.now();
        System.out.println("timeEnd : " + timeEnd);
        System.out.println("time : " + java.time.LocalTime.now().compareTo(time));
        System.out.println("time : " + time.compareTo(java.time.LocalTime.now()));
    }

}
