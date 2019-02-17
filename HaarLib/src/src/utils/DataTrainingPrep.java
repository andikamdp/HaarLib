/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package src.utils;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;
import src.entity.Data;

/**
 *
 * @author Andika Mulyawan
 */
public class DataTrainingPrep {

//######################################################################
    /**
     * method untuk menyiapkan label data training
     * param:
     * int i : jumlah data training
     * int label : label (class) data training
     * var:
     * Mat labelsMat : wadah label data training
     * int[] labels : wadah label untuk dimasukan ke labelsMat
     *
     * @param i
     * @param label
     * @return labelsMat
     */
    public static Mat getLabel(int i, int label) {
        int[] labels = {label};
        Mat labelsMat = new Mat(i, 1, CvType.CV_32SC1);
        for (int j = 0; j < i; j++) {
            labelsMat.put(j, 0, labels);
        }
        return labelsMat;
    }

    //######################################################################
    /**
     * method untuk memperoleh data training berdasarkan fitur garis tepi
     * param:
     * String lokasi : lokasi data training
     * List<Integer> index : indeks file untuk sampel
     * Boolean train : menentukan data untuk sampel atau training
     * var:
     * File folder : lokasi direktori data gambar
     * File[] listOfFiles : wadah array file gambar dalam folder
     * Mat trainingDataMat : wadah hasil deskripsi gambar dalam bentuk Mat
     *
     * @param lokasi
     * @param index
     * @param train
     * @return
     */
    public static List<Data> getDataSVMEdge(String lokasi, List<Integer> index, Boolean train) {
        File folder = new File(lokasi);
        File[] listOfFiles = folder.listFiles();
        List<Data> datas = new ArrayList<>();
        int row = 0;
        for (int i = 0; i < listOfFiles.length; i++) {
            if (!index.contains(i) && train) {
                float[] trainingData = getImageEdgeDescriptor(listOfFiles[i].getAbsolutePath());
//                Mat dataFile = new Mat(1, trainingData.length, CvType.CV_32FC1);
                Mat dataFile = new Mat(1, 64 * 48, CvType.CV_32FC1);
                System.out.println(trainingData.length);
                dataFile.put(0, 0, trainingData);
//                Mat dataFile = getImageEdgeDescriptorED(listOfFiles[i].getAbsolutePath());
                Data dataTraining = new Data(listOfFiles[i], dataFile, listOfFiles[i].getName(), i);
                datas.add(dataTraining);
                row++;
            } else if (index.contains(i) && !train) {
                float[] trainingData = getImageEdgeDescriptor(listOfFiles[i].getAbsolutePath());
//                Mat dataFile = new Mat(1, trainingData.length, CvType.CV_32FC1);
                Mat dataFile = new Mat(1, 64 * 48, CvType.CV_32FC1);
                dataFile.put(0, 0, trainingData);
//                Mat dataFile = getImageEdgeDescriptorED(listOfFiles[i].getAbsolutePath());
                Data dataTraining = new Data(listOfFiles[i], dataFile, listOfFiles[i].getName(), i);
                datas.add(dataTraining);
                row++;
            }
        }
        return datas;
    }

//######################################################################
    /**
     * method untuk memperoleh data training berdasarkan fitur garis tepi
     * param:
     * String lokasi : lokasi data training
     * var:
     * Mat hand : wadah data gambar
     * float[] trainingData : wadah nilai deskripsi gambar
     *
     * @param lokasi
     * @return
     */
    public static float[] getImageEdgeDescriptor(String lokasi) {
        Mat hand = Imgcodecs.imread(lokasi, CvType.CV_32F);
        hand = Preprocessing.getEdge(hand);
//        System.out.println(hand.rows() + " " + hand.cols());
        float[] trainingData = new float[hand.cols()];
        for (int j = 0; j < hand.cols(); j++) {
            trainingData[j] = (float) hand.get(0, j)[0];
        }
        return trainingData;
    }

    public static float[] getImageEdgeDescriptor(Mat hand) {
        hand = Preprocessing.getEdge(hand);
        float[] trainingData = new float[hand.cols()];
        for (int j = 0; j < hand.cols(); j++) {
            trainingData[j] = (float) hand.get(0, j)[0];
        }
        return trainingData;
    }

    public static Mat getImageEdgeDescriptorED(String lokasi) {
        Mat hand = Imgcodecs.imread(lokasi, CvType.CV_32F);
        hand = Preprocessing.getEdge(hand);
        return hand;
    }

    /**
     * //######################################################################
     * method untuk memeriksa memperoleh data training berdasarkan fitur HOG
     * var:
     * File folder : lokasi direktori data gambar
     * File[] listOfFiles :
     * Mat trainingDataMat :
     * Mat hand :
     * float[] trainingData:
     *
     * @param lokasi
     * @param index
     * @param train
     * @return
     *
     * public static List<Data> getDataSVMHog(String lokasi, List<Integer> index, Boolean train) {
     * File folder = new File(lokasi);
     * File[] listOfFiles = folder.listFiles();
     * List<Data> datas = new ArrayList<>();
     * int row = 0;
     * for (int i = 0; i < listOfFiles.length; i++) {
     * if (!index.contains(i) && train) {
     * Mat dataFile = new Mat(1, 192780, CvType.CV_32FC1);
     * dataFile.put(0, 0, getImageHogDescriptor(listOfFiles[i].getAbsolutePath()));
     * Data dataTraining = new Data(listOfFiles[i], dataFile, listOfFiles[i].getName(), i);
     * datas.add(dataTraining);
     * row++;
     * } else if (index.contains(i) && !train) {
     * Mat dataFile = new Mat(1, 192780, CvType.CV_32FC1);
     * dataFile.put(0, 0, getImageHogDescriptor(listOfFiles[i].getAbsolutePath()));
     * Data dataTraining = new Data(listOfFiles[i], dataFile, listOfFiles[i].getName(), i);
     * datas.add(dataTraining);
     * row++;
     * }
     * }
     * return datas;
     * }
     */
    /**
     * //######################################################################
     * method untuk memperoleh data training berdasarkan fitur HOG
     * param:
     * String lokasi : lokasi data training
     * var:
     * Mat hand : wadah data gambar
     * float[] trainingData : wadah nilai deskripsi gambar
     * HOGDescriptor gDescriptor : menampung class HOG
     * MatOfFloat descriptors : menampung hasil deskripsi image
     *
     * @param lokasi
     * @return
     *
     * public static float[] getImageHogDescriptor(String lokasi) {
     * HOGDescriptor gDescriptor = new HOGDescriptor();
     * Mat hand = Imgcodecs.imread(lokasi, CvType.CV_32F);
     * // Imgproc.resize(hand, hand, new Size(192, 144));
     * MatOfFloat descriptors = new MatOfFloat();
     * gDescriptor.compute(hand, descriptors);
     * float[] trainingData = descriptors.toArray();
     * // for (int j = 0; j < trainingData.length; j++) {
     * // trainingData[j] = Math.round(trainingData[j] * 100000) / 100;
     * // }
     * return trainingData;
     * }
     */
    //######################################################################
    /**
     *
     */
    public static Mat getDataMat(List<Data> data) {
        Mat dataMat = new Mat();
        for (Data data1 : data) {
            dataMat.push_back(data1.getDataMat());
        }
        return dataMat;
    }
}
