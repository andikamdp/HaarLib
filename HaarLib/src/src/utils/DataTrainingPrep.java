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
    public static Mat getDataSVMEdge(String lokasi, List<Integer> index, Boolean train) {
        File folder = new File(lokasi);
        File[] listOfFiles = folder.listFiles();
        Mat trainingDataMat;
        if (train) {
            trainingDataMat = new Mat(listOfFiles.length - index.size(), 48 * 64, CvType.CV_32FC1);
        } else {
            trainingDataMat = new Mat(index.size(), 48 * 64, CvType.CV_32FC1);
        }
        int row = 0;

        for (int i = 0; i < listOfFiles.length; i++) {
            if (!index.contains(i) && train) {
                trainingDataMat.put(row, 0, getImageEdgeDescriptor(listOfFiles[i].getAbsolutePath()));
                row++;
            } else if (index.contains(i) && !train) {
                trainingDataMat.put(row, 0, getImageEdgeDescriptor(listOfFiles[i].getAbsolutePath()));
                row++;
            }
        }
        return trainingDataMat;
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
        float[] trainingData = new float[hand.cols()];
        for (int j = 0; j < hand.cols(); j++) {
            trainingData[j] = (float) hand.get(0, j)[0];
        }
        return trainingData;
    }

//######################################################################
    /**
     * method untuk memeriksa memperoleh data training berdasarkan fitur HOG
     * var:
     * File folder : lokasi direktori data gambar
     * File[] listOfFiles :
     * Mat trainingDataMat :
     * Mat hand :
     * float[] trainingData:
     */
    public static Mat getDataSVMHog(String lokasi, List<Integer> index, Boolean train) {

        File folder = new File(lokasi);
        File[] listOfFiles = folder.listFiles();
        Mat trainingDataMat;
        if (train) {
            trainingDataMat = new Mat(listOfFiles.length - index.size(), 192780, CvType.CV_32FC1);
        } else {
            trainingDataMat = new Mat(index.size(), 192780, CvType.CV_32FC1);
        }
        int row = 0;
        for (int i = 0; i < listOfFiles.length; i++) {
            if (!index.contains(i) && train) {
                trainingDataMat.put(row, 0, getImageHogDescriptor(listOfFiles[i].getAbsolutePath()));
                row++;
            } else if (index.contains(i) && !train) {
                trainingDataMat.put(row, 0, getImageHogDescriptor(listOfFiles[i].getAbsolutePath()));
                row++;
            }
        }
        return trainingDataMat;
    }

//######################################################################
    /**
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
     */
    public static float[] getImageHogDescriptor(String lokasi) {
        HOGDescriptor gDescriptor = new HOGDescriptor();
        Mat hand = Imgcodecs.imread(lokasi, CvType.CV_32F);
        Imgproc.resize(hand, hand, new Size(192, 144));
        MatOfFloat descriptors = new MatOfFloat();
        gDescriptor.compute(hand, descriptors);
        float[] trainingData = descriptors.toArray();
        for (int j = 0; j < trainingData.length; j++) {
            trainingData[j] = Math.round(trainingData[j] * 100000) / 100;
        }
        return trainingData;
    }

//######################################################################
    /**
     * method untuk nama file dari data sampel dan training
     * param:
     * String lokasi : lokasi data training
     * List<Integer> index : indeks file untuk sampel
     * Boolean train : menentukan data untuk sampel atau training
     * var:
     * Mat hand :
     * List<String> folderName : wadah nama fariabel gambar
     * File folder : lokasi direktori data gambar
     * File[] listOfFiles : wadah array file gambar dalam folder
     *
     * @param lokasi
     * @param index
     * @param train
     * @return
     */
    public static List<String> getFileName(String lokasi, List<Integer> index, Boolean train) {
        List<String> folderName = new ArrayList<>();
        File folder = new File(lokasi);

        File[] listOfFiles = folder.listFiles();
        if (train) {
            for (int i = 0; i < listOfFiles.length; i++) {
                if (!index.contains(i)) {
                    folderName.add(listOfFiles[i].getName());
                }
            }
        } else {
            for (Integer listOfFile : index) {
                folderName.add(listOfFiles[listOfFile].getName());
                System.out.println(listOfFiles[listOfFile].getName());
            }
        }
        return folderName;
    }
}
