/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package src.controller;

import java.io.File;
import java.net.URL;
import java.util.ResourceBundle;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.TextArea;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
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
    private MainAppController mainAppController;

    /**
     * Initializes the controller class.
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
    }

    @FXML
    private void trainOnClick(ActionEvent event) {
//        SVMTry();
        svmTryBisindo();
    }

    @FXML
    private void predictOnAction(ActionEvent event) {
    }

//######################################################################
    /**
     * method untuk train SVM dengan data gambar BISINDO
     * var:
     * SVM svm : titik saat ini
     * Mat trainingDataMat :
     * Mat labelsMat :
     * Mat sampleDataMat :
     * int rows :
     * File folder :
     * File[] listOfFiles :
     * int[][] confusionMatrix :
     * float label :
     * int l,m :
     */
    public void svmTryBisindo() {
        SVM svm = SVM.create();
        svm.setType(SVM.C_SVC);
        svm.setKernel(SVM.LINEAR);
        svm.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER, 100, 1e-6));
        //######################################################################
        System.out.println("C_used");
        Mat trainingDataMat = getTrainSVMEdge("C_used");
        int rows = trainingDataMat.rows();
        Mat labelsMat = getLabel(rows, 12345);
        //######################################################################
        System.out.println("I_used");
        trainingDataMat.push_back(getTrainSVMEdge("I_used"));
        labelsMat.push_back(getLabel(rows, 15));
        //######################################################################
        System.out.println("L_used");
        trainingDataMat.push_back(getTrainSVMEdge("L_used"));
        labelsMat.push_back(getLabel(rows, 12));
        //######################################################################
        System.gc();
        //######################################################################
        System.out.println("O_used");
        trainingDataMat.push_back(getTrainSVMEdge("O_used"));
        labelsMat.push_back(getLabel(rows, 125));
        //######################################################################
        System.out.println("U_used");
        trainingDataMat.push_back(getTrainSVMEdge("U_used"));
        labelsMat.push_back(getLabel(rows, 123));
        //######################################################################
        System.out.println("V_used");
        trainingDataMat.push_back(getTrainSVMEdge("V_used"));
        labelsMat.push_back(getLabel(rows, 2));
        //######################################################################
        System.gc();
        //######################################################################
        svm.train(trainingDataMat, Ml.ROW_SAMPLE, labelsMat);
        svm.save("E:\\TA\\hCoba.xml");
        trainingDataMat = null;
        labelsMat = null;
        System.gc();
        //######################################################################
        Mat sampleDataMat = getTrainSVMEdge("Sample");
        File folder = new File("E:\\TA\\HandLearnSVM\\BISINDO\\Sample");
        File[] listOfFiles = folder.listFiles();
        int[][] confusionMatrix = new int[6][6];
        for (int j = 0; j < sampleDataMat.rows(); j++) {
            float label = svm.predict(sampleDataMat.row(j));
            int l = 0, m = 0;
            if (listOfFiles[j].getName().contains("C_")) {
                l = 0;
            } else if (listOfFiles[j].getName().contains("I_")) {
                l = 1;
            } else if (listOfFiles[j].getName().contains("L_")) {
                l = 2;
            } else if (listOfFiles[j].getName().contains("O_")) {
                l = 3;
            } else if (listOfFiles[j].getName().contains("U_")) {
                l = 4;
            } else if (listOfFiles[j].getName().contains("V_")) {
                l = 5;
            }
            if (label == 12345.0) {
                m = 0;
            } else if (label == 15.0) {
                m = 1;
            } else if (label == 12.0) {
                m = 2;
            } else if (label == 125.0) {
                m = 3;
            } else if (label == 123.0) {
                m = 4;
            } else if (label == 2.0) {
                m = 5;
            }
            confusionMatrix[l][m] += 1;
            System.out.println(listOfFiles[j].getName() + ": " + label + " " + l + ", " + m);

        }
        for (int i = 0; i < 6; i++) {
            for (int k = 0; k < 6; k++) {
                System.out.print(confusionMatrix[i][k] + " ");
            }
            System.out.println("");
        }
        svm.save("E:\\TA\\hCoba.xml");
    }
//######################################################################

    /**
     * method untuk train SVM dengan data gambar kombinasi jari terangkat
     * var:
     * SVM svm : titik saat ini
     * Mat trainingDataMat :
     * Mat labelsMat :
     * Mat sampleDataMat :
     * int rows :
     * File folder :
     * File[] listOfFiles :
     * int[][] confusionMatrix :
     * float label :
     * int l,m :
     */
    public void svmTry() {
        SVM svm = SVM.create();
        svm.setType(SVM.C_SVC);
        svm.setKernel(SVM.LINEAR);
        svm.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER, 100, 1e-6));
        //######################################################################
        System.out.println("Try1HandFull");
        Mat trainingDataMat = getTrainSVMEdge("Try1HandFull");
        Mat labelsMat = getLabel(trainingDataMat.rows(), 12345);
        //######################################################################
        System.out.println("Try1JempolKelingking");
        trainingDataMat.push_back(getTrainSVMEdge("Try1JempolKelingking"));
        int rows = trainingDataMat.rows();
        labelsMat.push_back(getLabel(rows, 15));
        //######################################################################
        System.out.println("Try1JempolTelunjuk");
        trainingDataMat.push_back(getTrainSVMEdge("Try1JempolTelunjuk"));
        labelsMat.push_back(getLabel(rows, 12));
        //######################################################################
        System.gc();
        //######################################################################
        System.out.println("Try1JempolTelunjukKelingking");
        trainingDataMat.push_back(getTrainSVMEdge("Try1JempolTelunjukKelingking"));
        labelsMat.push_back(getLabel(rows, 125));
//        trainingDataMat = getTrainSVMEdge("Try1JempolTelunjukKelingking");
//        labelsMat = getLabel(trainingDataMat.rows(), 125);
        //######################################################################
        System.out.println("Try1JempolTelunjukTengah");
        trainingDataMat.push_back(getTrainSVMEdge("Try1JempolTelunjukTengah"));
        labelsMat.push_back(getLabel(rows, 123));
        //######################################################################
        System.out.println("Try1telunjuk");
        trainingDataMat.push_back(getTrainSVMEdge("Try1telunjuk"));
        labelsMat.push_back(getLabel(rows, 2));
        //######################################################################
        System.gc();
        //######################################################################
        svm.train(trainingDataMat, Ml.ROW_SAMPLE, labelsMat);
        svm.save("E:\\TA\\hCoba.xml");
        trainingDataMat = null;
        labelsMat = null;
        System.gc();
        //######################################################################
        Mat sampleDataMat = getTrainSVMEdge("Sample");
        File folder = new File("E:\\TA\\HandLearnSVM\\Try1\\Sample");
        File[] listOfFiles = folder.listFiles();
        int[][] confusionMatrix = new int[6][6];
        for (int j = 0; j < sampleDataMat.rows(); j++) {
            float label = svm.predict(sampleDataMat.row(j));
            int l = 0, m = 0;
            if (listOfFiles[j].getName().contains("HandFull")) {
                l = 0;
            } else if (listOfFiles[j].getName().contains("JempolKelingking")) {
                l = 1;
            } else if (listOfFiles[j].getName().contains("JempolTelunjukKelingking")) {
                l = 2;
            } else if (listOfFiles[j].getName().contains("JempolTelunjukTengah")) {
                l = 3;
            } else if (listOfFiles[j].getName().contains("JempolTelunjuk")) {
                l = 4;
            } else if (listOfFiles[j].getName().contains("telunjuk")) {
                l = 5;
            }
            if (label == 12345.0) {
                m = 0;
            } else if (label == 15.0) {
                m = 1;
            } else if (label == 125.0) {
                m = 2;
            } else if (label == 123.0) {
                m = 3;
            } else if (label == 12.0) {
                m = 4;
            } else if (label == 2.0) {
                m = 5;
            }
            confusionMatrix[l][m] += 1;
            System.out.println(listOfFiles[j].getName() + ": " + label + " " + l + ", " + m);
        }
        for (int i = 0; i < 6; i++) {
            for (int k = 0; k < 6; k++) {
                System.out.print(confusionMatrix[i][k] + " ");
            }
            System.out.println("");
        }
        svm.save("E:\\TA\\hCoba.xml");
    }

//######################################################################
    /**
     * method untuk menyiapkan label data training
     * var:
     * int i : jumlah data training
     * int label : label (class) data training
     * Mat labelsMat : wadah label data training
     * int[] labels :
     */
    public Mat getLabel(int i, int label) {
        int[] labels = {label};
        Mat labelsMat = new Mat(i, 1, CvType.CV_32SC1);
        for (int j = 0; j < i; j++) {
            labelsMat.put(j, 0, labels);
        }
        return labelsMat;
    }

//######################################################################
    /**
     * method untuk memeriksa memperoleh data training berdasarkan fitur garis tepi
     * var:
     * File folder : lokasi direktori data gambar
     * File[] listOfFiles :
     * Mat trainingDataMat :
     * Mat hand :
     * float[] trainingData:
     */
    public Mat getTrainSVMEdge(String lokasi) {
//        File folder = new File("E:\\TA\\HandLearnSVM\\Try1\\" + lokasi);
        File folder = new File("E:\\TA\\HandLearnSVM\\BISINDO\\" + lokasi);
        File[] listOfFiles = folder.listFiles();
        Mat trainingDataMat = new Mat(listOfFiles.length, 48 * 64, CvType.CV_32FC1);
        System.out.println(listOfFiles.length);
        System.out.println(folder.getPath());

        for (int i = 0; i < listOfFiles.length; i++) {
            Mat hand = Imgcodecs.imread(folder.getPath() + "\\" + listOfFiles[i].getName(), CvType.CV_32F);
            hand = Preprocessing.getEdge(hand);
            float[] trainingData = new float[hand.cols()];
            for (int j = 0; j < hand.cols(); j++) {
                trainingData[j] = (float) hand.get(0, j)[0];
            }
            trainingDataMat.put(i, 0, trainingData);
        }
        return trainingDataMat;
    }

//######################################################################
    /**
     * method untuk memeriksa apakah titik saat ini lebih tinggi dari titik sebelumnya
     * semakin kecil nilai titik pada frame semakin tinggi
     * var:
     * Point titikA : titik saat ini
     * Point titikB : titik sebelumnya
     *
     */
    void setMainController(MainAppController aThis) {
        mainAppController = aThis;
    }

}
