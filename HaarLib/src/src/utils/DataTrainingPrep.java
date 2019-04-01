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
import org.opencv.imgcodecs.Imgcodecs;
import src.entity.Data;

/**
 *
 * @author Andika Mulyawan
 */
public final class DataTrainingPrep {

    /**
     * ######################################################################
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

    /**
     * ######################################################################
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
    public static List<Data> getDataSVMEdge(String lokasi, List<Integer> index, Boolean train, double width, double treshold) {
        File folder = new File(lokasi);
        File[] listOfFiles = folder.listFiles();
        List<Data> datas = new ArrayList<>();
        int row = 0;
        for (int i = 0; i < listOfFiles.length; i++) {
            if (!index.contains(i) && train) {
                Mat dataFile = getImageEdgeDescriptor(listOfFiles[i].getAbsolutePath(), width, treshold);
                Data dataTraining = new Data(listOfFiles[i], dataFile, listOfFiles[i].getName(), i);
                datas.add(dataTraining);
                row++;
            } else if (index.contains(i) && !train) {
                Mat dataFile = getImageEdgeDescriptor(listOfFiles[i].getAbsolutePath(), width, treshold);
                Data dataTraining = new Data(listOfFiles[i], dataFile, listOfFiles[i].getName(), i);
                datas.add(dataTraining);
                row++;
            }
        }
        return datas;
    }

    /**
     * ######################################################################
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
    public static Mat getImageEdgeDescriptor(String lokasi, double width, double treshold) {
        Mat hand = Imgcodecs.imread(lokasi, CvType.CV_32FC1);
        Mat trainingDataMat;
        double height = Preprocessing.getHeight(width, hand.width(), hand.height());
        hand = Preprocessing.getEdge(hand, width, height, treshold);
        float[] trainingData = new float[hand.cols()];
        trainingDataMat = new Mat(1, hand.cols(), CvType.CV_32F);
        for (int j = 0; j < hand.cols(); j++) {
            trainingData[j] = (float) hand.get(0, j)[0];
        }
        trainingDataMat.put(0, 0, trainingData);
        return trainingDataMat;
    }

    public static Mat getImageEdgeDescriptor(Mat hand, double width, double treshold) {
        Mat trainingDataMat;
        double height = Preprocessing.getHeight(width, hand.width(), hand.height());
        hand = Preprocessing.getEdge(hand, width, height, treshold);
        float[] trainingData = new float[hand.cols()];
        trainingDataMat = new Mat(1, hand.cols(), CvType.CV_32F);
        for (int j = 0; j < hand.cols(); j++) {
            trainingData[j] = (float) hand.get(0, j)[0];
        }
        trainingDataMat.put(0, 0, trainingData);
        return trainingDataMat;
    }

    /**
     * ######################################################################
     *
     * @param data
     * @return
     */
    public static Mat getDataMat(List<Data> data) {
        Mat dataMat = new Mat();
        for (Data data1 : data) {
            dataMat.push_back(data1.getDataMat());
        }
        return dataMat;
    }

}
