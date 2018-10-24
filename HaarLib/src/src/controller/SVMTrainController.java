/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package src.controller;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.ResourceBundle;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.TextArea;
import javafx.scene.image.Image;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import org.opencv.objdetect.HOGDescriptor;
import src.Utils;
import src.utils.Preprocessing;

/**
 * FXML Controller class
 *
 * @author Andika Mulyawan
 */
public class SVMTrainController implements Initializable {

    @FXML
    private Button btnTrain;
    @FXML
    private Button btnPredict;
    @FXML
    private TextArea txtAreaStatus;
///////////////////////////////////
    private MainAppController mainAppController;
    List<Integer> devContourIdxList;
    List<MatOfPoint> contous;
    List<MatOfInt4> devOfInt4s;
    List<MatOfPoint> devOfPoints;
    List<Integer> Puncak = new ArrayList<>();
    List<Integer> Lembah = new ArrayList<>();

    /**
     * Initializes the controller class.
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
    }

    @FXML
    private void trainOnClick(ActionEvent event) {
        SVMTry();
    }

    @FXML
    private void predictOnAction(ActionEvent event) {
    }

    public void SVMTry() {
        SVM svm = SVM.create();
        svm.setType(SVM.C_SVC);
        svm.setKernel(SVM.LINEAR);
        svm.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER, 100, 1e-6));
        //######################################################################
        System.out.println("Try1HandFull");
        Mat trainingDataMat = getTrainSVMEdge("Try1HandFull");
        Mat labelsMat = getLabel(trainingDataMat.rows(), 12345);
//        trainingDataMat.push_back(trainingDataMat);
//        labelsMat.push_back(labelsMat);
        //######################################################################
        System.out.println("Try1JempolKelingking");
        Mat trainingDataMat_2 = getTrainSVMEdge("Try1JempolKelingking");
        Mat labelsMat_2 = getLabel(trainingDataMat_2.rows(), 15);
        trainingDataMat.push_back(trainingDataMat_2);
        labelsMat.push_back(labelsMat_2);
        //######################################################################
        System.out.println("Try1JempolTelunjuk");
        Mat trainingDataMat_3 = getTrainSVMEdge("Try1JempolTelunjuk");
        Mat labelsMat_3 = getLabel(trainingDataMat_3.rows(), 12);
        trainingDataMat.push_back(trainingDataMat_3);
        labelsMat.push_back(labelsMat_3);
        //######################################################################
        svm.train(trainingDataMat, Ml.ROW_SAMPLE, labelsMat);
        trainingDataMat = null;
        labelsMat = null;
        System.gc();
        //######################################################################
        System.out.println("Try1JempolTelunjukKelingking");
//        Mat trainingDataMat_4 = getTrainSVMEdge("Try1JempolTelunjukKelingking");
//        Mat labelsMat_4 = getLabel(trainingDataMat_4.rows(), 125);
//        trainingDataMat.push_back(trainingDataMat_4);
//        labelsMat.push_back(labelsMat_4);
        trainingDataMat = getTrainSVMEdge("Try1JempolTelunjukKelingking");
        labelsMat = getLabel(trainingDataMat.rows(), 125);
        //######################################################################
        System.out.println("Try1JempolTelunjukTengah");
        Mat trainingDataMat_5 = getTrainSVMEdge("Try1JempolTelunjukTengah");
        Mat labelsMat_5 = getLabel(trainingDataMat_5.rows(), 123);
        trainingDataMat.push_back(trainingDataMat_5);
        labelsMat.push_back(labelsMat_5);
        //######################################################################
        System.out.println("Try1telunjuk");
        Mat trainingDataMat_6 = getTrainSVMEdge("Try1telunjuk");
        Mat labelsMat_6 = getLabel(trainingDataMat_6.rows(), 2);
        trainingDataMat.push_back(trainingDataMat_6);
        labelsMat.push_back(labelsMat_6);
        //######################################################################
        svm.train(trainingDataMat, Ml.ROW_SAMPLE, labelsMat);
        trainingDataMat = null;
        labelsMat = null;
        System.gc();
        //######################################################################
        System.out.println("Try1TelunjukKelingking");
//        Mat trainingDataMat_7 = getTrainSVMEdge("Try1TelunjukKelingking");
//        Mat labelsMat_7 = getLabel(trainingDataMat_7.rows(), 25);
//        trainingDataMat.push_back(trainingDataMat_7);
//        labelsMat.push_back(labelsMat_7);
        trainingDataMat = getTrainSVMEdge("Try1TelunjukKelingking");
        labelsMat = getLabel(trainingDataMat.rows(), 25);
        //######################################################################
        System.out.println("Try1telunjukTengah");
        Mat trainingDataMat_8 = getTrainSVMEdge("Try1telunjukTengah");
        Mat labelsMat_8 = getLabel(trainingDataMat_8.rows(), 23);
        trainingDataMat.push_back(trainingDataMat_8);
        labelsMat.push_back(labelsMat_8);
        //################svm.train(trainingDataMat, Ml.ROW_SAMPLE, labelsMat);######################################################
        svm.train(trainingDataMat, Ml.ROW_SAMPLE, labelsMat);
        trainingDataMat = null;
        labelsMat = null;
        System.gc();
        //######################################################################
        svm.train(trainingDataMat, Ml.ROW_SAMPLE, labelsMat);
        svm.save("E:\\TA\\hCoba.xml");
        //######################################################################
//        List<Mat> sampleDataMat = getSampleSVM("penuhSample");
//        System.out.println("trainingDataMat.rows() " + sampleDataMat.size());
//        for (Mat mat : sampleDataMat) {
//            float label = svm.predict(mat);
//            System.out.println("label " + label);
//        }
        Mat sampleDataMat = getSampleSVMEdge("a");
        File folder = new File("E:\\TA\\HandLearnSVM\\Try1\\Sample");
        File[] listOfFiles = folder.listFiles();
        for (int j = 0; j < sampleDataMat.rows(); j++) {
            float label = svm.predict(sampleDataMat.row(j));
            System.out.println(listOfFiles[j].getName() + ": " + label);
        }
        svm.save("E:\\TA\\hCoba.xml");
    }

    public Mat getTrainSVM(String lokasi) {
/////////////
//get image from file to arr of file
/////////////
        File folder = new File("E:\\TA\\HandLearnSVM\\" + lokasi);
        File[] listOfFiles = folder.listFiles();
/////////////
//prepate Mat
/////////////
        Mat trainingDataMat = new Mat(listOfFiles.length, 10, CvType.CV_32FC1);

//****************************************************************************//
//****************************************************************************//
//****************************************************************************//
/////////////
//
/////////////
        for (int i = 0; i < listOfFiles.length; i++) {
            /////////////
            //
            /////////////
            Mat hand = Imgcodecs.imread("E:\\TA\\HandLearnSVM\\" + lokasi + "\\"
                    + listOfFiles[i].getName());
            System.out.println("E:\\TA\\HandLearnSVM\\" + lokasi + "\\" + listOfFiles[i].getName());

            Start_Try(hand);

//        drawJumlahJari(contous, hand, Puncak);
            float[] trainingData = new float[Puncak.size() * 2];
            int k = 0;
            Point[] contourPoint = contous.get(0).toArray();
            System.out.println("train SVM");
            System.out.println(Puncak.size());
            for (int j = 0; j < Puncak.size(); j++) {
                System.out.println(Puncak.get(j));
                trainingData[k] = (float) contourPoint[Puncak.get(j)].x;
                trainingData[k + 1] = (float) contourPoint[Puncak.get(j)].y;
                k += 2;
            }
            trainingDataMat.put(i, 0, trainingData);
//
        }
        return trainingDataMat;
    }

    public List<Mat> getSampleSVM(String lokasi) {
        File folder = new File("E:\\TA\\HandLearnSVM\\" + lokasi);
        File[] listOfFiles = folder.listFiles();
//
        List<Mat> sampleList = new ArrayList<>();
        for (int i = 0; i < listOfFiles.length; i++) {
            Mat sampel = new Mat(1, 10, CvType.CV_32FC1);
            Mat hand = Imgcodecs.imread("E:\\TA\\HandLearnSVM\\" + lokasi + "\\"
                    + listOfFiles[i].getName());
            Start_Try(hand);

//        drawJumlahJari(contous, hand, Puncak);
            float[] trainingData = new float[Puncak.size() * 2];
            int k = 0;
            Point[] contourPoint = contous.get(0).toArray();
            System.out.println("train SVM");
            System.out.println(Puncak.size());
            for (int j = 0; j < Puncak.size(); j++) {
                System.out.println(Puncak.get(j));
                trainingData[k] = (float) contourPoint[Puncak.get(j)].x;
                trainingData[k + 1] = (float) contourPoint[Puncak.get(j)].y;
                k += 2;
            }
            sampel.put(0, 0, trainingData);
            sampleList.add(sampel);
        }
        for (File listOfFile : listOfFiles) {
            System.out.println(listOfFile.getName());
        }
//
        return sampleList;
    }

    public Mat getLabel(int i, int label) {
        int[] labels = {label};
        Mat labelsMat = new Mat(i, 1, CvType.CV_32SC1);
        for (int j = 0; j < i; j++) {
            labelsMat.put(j, 0, labels);
        }
        return labelsMat;
    }

    public Mat getTrainSVMEdge(String lokasi) {
        File folder = new File("E:\\TA\\HandLearnSVM\\Try1\\" + lokasi);
//        File folder = new File("E:\\TA\\HandLearnSVM\\leapGestRecog\\train\\" + lokasi);
//        File folder = new File("E:\\TA\\HandLearnSVM\\hand-gestures\\t\\" + lokasi);
        File[] listOfFiles = folder.listFiles();
/////////////
//prepate Mat
/////////////
        Mat trainingDataMat = new Mat(listOfFiles.length, 480 * 640, CvType.CV_32FC1);
        System.out.println(listOfFiles.length);
        System.out.println(folder.getPath());

        for (int i = 0; i < listOfFiles.length; i++) {
            Mat hand = Imgcodecs.imread(folder.getPath() + "\\" + listOfFiles[i].getName(), CvType.CV_32F);

            hand = getEdge(hand);
//            System.out.println("hand " + hand.rows() + " " + hand.cols());
            float[] trainingData = new float[hand.cols()];
            for (int j = 0; j < hand.cols(); j++) {
                trainingData[j] = (float) hand.get(0, j)[0];
            }

            trainingDataMat.put(i, 0, trainingData);

        }
        return trainingDataMat;
    }

    private void Start_Try(Mat frame) {
        Puncak = new ArrayList<>();
        Image imageToMat;
        //# flip the frame so that it is not the mirror view
        Core.flip(frame, frame, 1);
        Preprocessing.drawRect(frame);
        Mat hand = Preprocessing.getBox(frame.clone());
        /////////////////////////////////
        //memperoleh segment image biner
        Mat tresholded = Preprocessing.segment(hand.clone());
        //////////////////////////////////////
        contous = Preprocessing.getContour(tresholded);
        devOfInt4s = Preprocessing.getDevectIndexPoint(contous);
        Preprocessing.toListMatOfPointDevec(contous, devOfInt4s, devContourIdxList);
//                        HandRec(contous, hand);
//                        hitungJarakTitik(devOfPoints);
        hapusTitik(contous, Preprocessing.getBox(frame));
//                        drawJumlahJari(contous, frame, Puncak);

    }

    public Mat getSampleSVMEdge(String lokasi) {
        File folder = new File("E:\\TA\\HandLearnSVM\\Try1\\Sample");
//        File folder = new File("E:\\TA\\HandLearnSVM\\leapGestRecog\\sample");
//        File folder = new File("E:\\TA\\HandLearnSVM\\hand-gestures\\s");
        File[] listOfFiles = folder.listFiles();
/////////////
//prepate Mat
/////////////
        Mat trainingDataMat = new Mat(listOfFiles.length, 480 * 640, CvType.CV_32FC1);

        System.out.println(listOfFiles.length);

        for (int i = 0; i < listOfFiles.length; i++) {
            Mat hand = Imgcodecs.imread(folder.getPath() + "\\" + listOfFiles[i].getName(), CvType.CV_32F);

            hand = getEdge(hand);
//            System.out.println("hand " + hand.rows() + " " + hand.cols());
            float[] trainingData = new float[hand.cols()];
            for (int j = 0; j < hand.cols(); j++) {
                trainingData[j] = (float) hand.get(0, j)[0];
            }
            System.out.println("trainingData.length " + trainingData.length);
            trainingDataMat.put(i, 0, trainingData);

        }
        return trainingDataMat;
    }

    private Mat getEdge(Mat frame) {
        frame = segment(frame);
//######################################################################
//        System.out.println("frame.channels() " + frame.channels());
//        System.out.println("frame.rows() " + frame.rows());
//        System.out.println("frame.cols() " + frame.cols());
//        frame = frame.reshape(1, 1);
//        System.out.println("frame.channels() " + frame.channels());
//        System.out.println("frame.rows() " + frame.rows());
//        System.out.println("frame.cols() " + frame.cols());
//######################################################################
        Core.flip(frame, frame, 1);
//######################################################################
//        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);

        Mat dist = new Mat();
//        batas minimum treshold
        Imgproc
                .threshold(frame, frame, 100, 255,
                        Imgproc.THRESH_BINARY_INV);

        frame = Preprocessing.cleaning(frame);
//######################################################################
        Imgproc.resize(frame, frame, new Size(640, 480));
//        Imgproc.Canny(frame, frame, 0.2, 0.2);
        frame = frame.reshape(1, 1);

//        Imgproc.resize(frame, frame, new Size(640 * 480, 1));
        return frame;
    }

//######################################################################
/////////////
//method untuk memisahkan objek dengan background
//kendala background masih harus bersih dan memiliki warna tidak selaras kulit
/////////////
    private Mat segment(Mat frameAsli) {
        try {
            double tres = 50.0;
//        Mat frameUbah = frameAsli.clone();
//            Scalar upperb = new Scalar(64, 223, 255);
//            Scalar lowerb = new Scalar(0, 0, 0);
//            Core.inRange(frameAsli, lowerb, upperb, frameAsli);
//            updateImageView(layarBW, Utils.mat2Image(frameAsli));
            Imgproc.cvtColor(frameAsli, frameAsli, Imgproc.COLOR_BGR2GRAY);

            Mat dist = new Mat();
//        batas minimum treshold
            Imgproc
                    .threshold(frameAsli, frameAsli, 100, 255,
                            Imgproc.THRESH_BINARY_INV);

            frameAsli = Preprocessing.cleaning(frameAsli);
        } catch (Exception e) {
            System.out.println("segment(Mat frameAsli)");
            System.out.println(e);
            System.out.println("");
        }

        return frameAsli;
    }
//######################################################################
//######################################################################

    void setMainController(MainAppController aThis) {
        mainAppController = aThis;
    }

    private void hapusTitik(List<MatOfPoint> contours, Mat hand) {
        try {
            Puncak = new ArrayList<>();
            Point[] point = contous.get(0).toArray();
            Puncak.addAll(devContourIdxList);
            Lembah.addAll(devContourIdxList);
//        System.out.println("isi index puncak awal");
//        for (Integer integer : Puncak) {
//            System.out.println(integer);
//        }
            //jika posisi false berarti cari lembah
            //jika posisi true berarti cari puncak
            Boolean puncak = true;
            for (int j = 0; j < Puncak.size(); j++) {
                int index = Puncak.get(j);
                int indexP = 0;
                if (j + 1 < Puncak.size()) {
                    indexP = Puncak.get(j + 1);
                }

                if (index < point.length && indexP < point.length
                        && point[index].y
                        < hand.rows() - 1
                        && point[indexP].y < hand.rows() - 1) {
                    if (puncak) {
                        //jika menemukan puncak index dicaatat
                        if (Preprocessing.arahTitikY(point[index], point[indexP])) {
                            puncak = false;
                            Lembah.set(j, -1);
                        } //jika titik lebih tinggi index sebelumnya dihapus
                        else {
                            Lembah.set(j, -1);
                            Puncak.set(j, -1);
                            devContourIdxList.set(j, -1);
                        }
                    } else {
                        //jika menemukan lembah index dicaatat
                        if (Preprocessing.arahTitikY(point[indexP], point[index])) {
                            Puncak.set(j, -1);
                            puncak = true;
                        } //jika titik lebih tinggi index sebelumnya dihapus
                        else {
                            Lembah.set(j, -1);
                            Puncak.set(j, -1);
                            devContourIdxList.set(j, -1);
                        }
                    }
                } else {
                    Puncak.set(j, -1);
                    Lembah.set(j, -1);
                    devContourIdxList.set(j, -1);
                }
            }
            Integer rem = -1;
//            Puncak.addAll(Lembah);
            Collections.sort(Puncak);

            for (Integer integer : Puncak) {
                System.out.println(integer);
            }
            System.out.println("");
            System.out.println("contous " + contous.get(0).toArray().length);
            while (Puncak.contains(rem)) {
                Puncak.remove(rem);
            }
            System.out.println("");
            for (Integer integer : Puncak) {
                System.out.println(integer);
            }
//        System.out.println("isi index puncak akhir");
//        for (Integer integer : Puncak) {
//            System.out.println(integer);
//        }
//        while (devContourIdxList.contains(rem)) {
//            devContourIdxList.remove(rem);
//        }
            Mat hand2 = hand.clone();
            Preprocessing.drawPointColor(contous, hand2, Puncak);
            Preprocessing.drawJumlahJari(hand, Puncak.size());
//            layarBW.setImage(Utils.mat2Image(hand2));
//            drawPointColor(contous, hand, devContourIdxList);
//            layarEdge.setImage(Utils.mat2Image(hand2));
        } catch (Exception e) {
            System.out.
                    println("hapusTitik(List<MatOfPoint> contours, Mat hand)");
            System.out.println(e);
            System.out.println("");
        }

    }

    public Mat getTrainSVMHOG(String lokasi) {
        File folder = new File("E:\\TA\\HandLearnSVM\\Try1\\" + lokasi);
        File[] listOfFiles = folder.listFiles();
/////////////
//prepate Mat
/////////////
        Mat trainingDataMat = new Mat(listOfFiles.length, 480 * 640, CvType.CV_32FC1);
        System.out.println(listOfFiles.length);
        System.out.println(folder.getPath());
        HOGDescriptor hog = new HOGDescriptor();
        MatOfFloat descriptors = new MatOfFloat();
        for (int i = 0; i < listOfFiles.length; i++) {
            Mat hand = Imgcodecs.imread(folder.getPath() + "\\" + listOfFiles[i].getName(), CvType.CV_32F);
            Imgproc.cvtColor(hand, hand, Imgproc.COLOR_BGR2GRAY);
            hog.compute(hand, descriptors);

            float[] trainingData = new float[descriptors.rows()];
            for (int j = 0; j < hand.cols(); j++) {
                trainingData[j] = (float) hand.get(j, 0)[0];
            }

            trainingDataMat.put(i, 0, trainingData);

        }
        return trainingDataMat;
    }
}
