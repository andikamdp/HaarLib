/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package src.controller;

import java.io.File;
import java.net.URL;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Observable;
import java.util.Random;
import java.util.ResourceBundle;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.TextArea;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import org.opencv.objdetect.HOGDescriptor;
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
    @FXML
    private ComboBox<String> cmbType;
//
    private int ratio;
    private MainAppController mainAppController;
    private SVM svm;
    private int seed;
    private List<Double> akurasiSeedSample;
    private List<Double> akurasiSeedTrain;
    private LocalTime time;
//

    /**
     * Initializes the controller class.
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO

        ratio = 30;
        ObservableList<String> type = FXCollections.observableArrayList();
        type.add("SVM");
        type.add("SVM Bisindo");
        type.add("SVM Sampel Acak");
        type.add("SVM Bisindo Sampel Acak");
        type.add("SVM Hog Sampel Acak");
        cmbType.setItems(type);
        svm = SVM.create();
//        svm.setC(0);
//        svm.setClassWeights(val);
//        svm.setCoef0(ratio);//poly, sigmoid
//        svm.setDegree(10.0);//poly
//        svm.setGamma(20.0);//poly, sigmoid, RBF, CHI2
        svm.setKernel(SVM.LINEAR);
//        svm.setNu(2);//NU_SVC, ONE_CLASS, NU_SVR
//        svm.setP(ratio);// SVM::EPS_SVR
//        svm.setTermCriteria(new TermCriteria(1, 10000, 10000));
//        svm.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER, 10000, 1e-6));
        System.out.println("\n" + "\nTermCriteria.MAX_ITER : " + TermCriteria.MAX_ITER + "\nTermCriteria.COUNT : " + TermCriteria.COUNT + "\nTermCriteria.EPS : " + TermCriteria.EPS + "\n");
        svm.setType(SVM.C_SVC);
//        svm.set
        seed = 0;
        akurasiSeedTrain = new ArrayList<Double>();
        akurasiSeedSample = new ArrayList<Double>();
    }

    @FXML
    private void trainOnClick(ActionEvent event) {
        akurasiSeedTrain.clear();
        akurasiSeedSample.clear();
        if (cmbType.getValue().equals("SVM")) {
            txtAreaStatus.setText(txtAreaStatus.getText() + "Train SVM Sampel Terpisah \n");
            svmTry();
        } else if (cmbType.getValue().equals("SVM Bisindo")) {
            txtAreaStatus.setText(txtAreaStatus.getText() + "Train SVM BISINDO Sampel Terpisah \n");
            svmTryBisindo();
        } else if (cmbType.getValue().equals("SVM Sampel Acak")) {
            txtAreaStatus.setText(txtAreaStatus.getText() + "Train SVM Sampel Acak \n");
            for (int i = 0; i < 10; i++) {
                seed = i;
                svmRandom();
            }
            rataRataAkurasiSeed();
        } else if (cmbType.getValue().equals("SVM Bisindo Sampel Acak")) {
            txtAreaStatus.setText(txtAreaStatus.getText() + "Train SVM BISINDO Sampel Acak \n");
            for (int i = 0; i < 10; i++) {
                seed = i;
                svmBisindoRandom();
            }
            rataRataAkurasiSeed();
        } else if (cmbType.getValue().equals("SVM Hog Sampel Acak")) {
            txtAreaStatus.setText(txtAreaStatus.getText() + "Train SVM Hog Sampel Acak \n");
//            for (int i = 0; i < 10; i++) {
            time = java.time.LocalTime.now();
            System.out.println("time : " + time);
            seed = 0;
            svmHogRandom();
//            }
            rataRataAkurasiSeed();
        }

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

        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "C_used \n");
        Mat trainingDataMat = getTrainSVMEdge("C_used");
        int rows = trainingDataMat.rows();
        Mat labelsMat = getLabel(rows, 12345);
        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "I_used \n");
        trainingDataMat.push_back(getTrainSVMEdge("I_used"));
        labelsMat.push_back(getLabel(rows, 15));
        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "L_used \n");
        trainingDataMat.push_back(getTrainSVMEdge("L_used"));
        labelsMat.push_back(getLabel(rows, 12));
        //######################################################################
        System.gc();
        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "O_used \n");
        trainingDataMat.push_back(getTrainSVMEdge("O_used"));
        labelsMat.push_back(getLabel(rows, 125));
        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "U_used \n");
        trainingDataMat.push_back(getTrainSVMEdge("U_used"));
        labelsMat.push_back(getLabel(rows, 123));
        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "V_used \n");
        trainingDataMat.push_back(getTrainSVMEdge("V_used"));
        labelsMat.push_back(getLabel(rows, 2));
        System.out.println("papapap trainingDataMat.rows() " + trainingDataMat.rows() + "trainingDataMat.cols() " + trainingDataMat.cols());
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
            txtAreaStatus.setText(txtAreaStatus.getText() + listOfFiles[j].getName() + ": " + label + " " + l + ", " + m + " \n");

        }
        for (int i = 0; i < 6; i++) {
            for (int k = 0; k < 6; k++) {
                txtAreaStatus.setText(txtAreaStatus.getText() + confusionMatrix[i][k] + " ");
            }
            txtAreaStatus.setText(txtAreaStatus.getText() + " \n");
        }
        confusionMatriks(confusionMatrix, sampleDataMat.rows(), false);
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

        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "Try1HandFull \n");
        Mat trainingDataMat = getTrainSVMEdge("Try1HandFull");
        int rows = trainingDataMat.rows();
        Mat labelsMat = getLabel(trainingDataMat.rows(), 12345);
        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "Try1JempolKelingking \n");
        trainingDataMat.push_back(getTrainSVMEdge("Try1JempolKelingking"));
        labelsMat.push_back(getLabel(rows, 15));
        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "Try1JempolTelunjuk \n");
        trainingDataMat.push_back(getTrainSVMEdge("Try1JempolTelunjuk"));
        labelsMat.push_back(getLabel(rows, 12));
        //######################################################################
        System.gc();
        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "Try1JempolTelunjukKelingking \n");
        trainingDataMat.push_back(getTrainSVMEdge("Try1JempolTelunjukKelingking"));
        labelsMat.push_back(getLabel(rows, 125));
        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "Try1JempolTelunjukTengah \n");
        trainingDataMat.push_back(getTrainSVMEdge("Try1JempolTelunjukTengah"));
        labelsMat.push_back(getLabel(rows, 123));
        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "Try1telunjuk \n");
        trainingDataMat.push_back(getTrainSVMEdge("Try1telunjuk"));
        labelsMat.push_back(getLabel(rows, 2));
        //######################################################################
        System.gc();
        //######################################################################
//        svm.train(trainingDataMat, Ml.ROW_SAMPLE, labelsMat);
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
            txtAreaStatus.setText(txtAreaStatus.getText() + listOfFiles[j].getName() + ": " + label + " " + l + ", " + m + "\n");
        }
        for (int i = 0; i < 6; i++) {
            for (int k = 0; k < 6; k++) {
                txtAreaStatus.setText(txtAreaStatus.getText() + confusionMatrix[i][k] + " ");
            }
            txtAreaStatus.setText(txtAreaStatus.getText() + " \n");
        }
        confusionMatriks(confusionMatrix, sampleDataMat.rows(), false);
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
        File folder;
        if (cmbType.getValue().contains("Bisindo")) {
            folder = new File("E:\\TA\\HandLearnSVM\\BISINDO\\" + lokasi);
        } else {
            folder = new File("E:\\TA\\HandLearnSVM\\Try1\\" + lokasi);
        }
        File[] listOfFiles = folder.listFiles();
        Mat trainingDataMat = new Mat(listOfFiles.length, 48 * 64, CvType.CV_32FC1);
        txtAreaStatus.setText(txtAreaStatus.getText() + listOfFiles.length + " \n");
        txtAreaStatus.setText(txtAreaStatus.getText() + folder.getPath() + " \n");

        for (int i = 0; i < listOfFiles.length; i++) {
            Mat hand = Imgcodecs.imread(folder.getPath() + "\\" + listOfFiles[i].getName(), CvType.CV_32F);
            Preprocessing.orb(hand.clone());
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
    public Mat getDataSVMEdgeDELETE(String lokasi, List<Integer> index, Boolean train) {
        File folder;
        if (cmbType.getValue().contains("Bisindo")) {
            folder = new File("E:\\TA\\HandLearnSVM\\BISINDO\\" + lokasi);
        } else {
            folder = new File("E:\\TA\\HandLearnSVM\\Try1\\" + lokasi);
        }
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
                Mat hand = Imgcodecs.imread(folder.getPath() + "\\" + listOfFiles[i].getName(), CvType.CV_32F);
                Preprocessing.orb(hand.clone());
                hand = Preprocessing.getEdge(hand);

                float[] trainingData = new float[hand.cols()];
                for (int j = 0; j < hand.cols(); j++) {
                    trainingData[j] = (float) hand.get(0, j)[0];
                }
                trainingDataMat.put(row, 0, trainingData);
                row++;
//                System.out.println("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq");
//                System.out.println("");
//                for (int j = 0; j < trainingDataMat.cols(); j += 200) {
//                    if (j < trainingDataMat.cols() - 1) {
//                        System.out.println("trainingDataMat.get(" + i + ", " + j + ") " + trainingDataMat.get(i, j)[0]);
//                    }
//                }
            } else if (index.contains(i) && !train) {
                Mat hand = Imgcodecs.imread(folder.getPath() + "\\" + listOfFiles[i].getName(), CvType.CV_32F);
                Preprocessing.orb(hand.clone());
                hand = Preprocessing.getEdge(hand);

                float[] trainingData = new float[hand.cols()];
                for (int j = 0; j < hand.cols(); j++) {
                    trainingData[j] = (float) hand.get(0, j)[0];
                }
                trainingDataMat.put(row, 0, trainingData);
                row++;
            }
        }
//        System.out.println("PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP");
//        for (int j = 0; j < trainingDataMat.rows(); j += 2000) {
//            for (int k = 0; k < trainingDataMat.cols(); k++) {
//                System.out.println("trainingDataMat.get(" + j + ", " + k + ").length :" + trainingDataMat.get(j, k)[0]);
//            }
//            System.out.println("");
//        }
        return trainingDataMat;
    }

    public List<String> getFileName(String lokasi, List<Integer> index, Boolean train) {
        List<String> folderName = new ArrayList<>();
        File folder;
        if (cmbType.getValue().contains("Bisindo")) {
            folder = new File("E:\\TA\\HandLearnSVM\\BISINDO\\" + lokasi);
        } else {
            folder = new File("E:\\TA\\HandLearnSVM\\Try1\\" + lokasi);
        }
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
            }
        }

        return folderName;
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
    public List<Integer> getRandomIndex(int jumlahData) {
        List<Integer> index = new ArrayList<>();
        List<Integer> indexSample = new ArrayList<>();
        for (int i = 0; i < jumlahData; i++) {
            index.add(i);
        }
        Random rand = new Random(seed);

//        int numberOfElements = ratio;
        int numberOfElements = (int) (((double) ratio / 100.0) * (double) jumlahData);;
        System.out.println("Daftar Random Index :" + numberOfElements);
        for (int i = 0; i < numberOfElements; i++) {
            int randomIndex = rand.nextInt(index.size());
            System.out.print(randomIndex + " ");
            indexSample.add(index.get(randomIndex));
            index.remove(randomIndex);
        }
        Collections.sort(indexSample);
//        System.out.println("#######################################################");
//        for (int i = 0; i < indexSample.size(); i++) {
//            System.out.println(indexSample.get(i));
//        }
//        System.out.println("#######################################################");
//        for (int i = 0; i < index.size(); i++) {
//            System.out.println(index.get(i));
//        }
        System.out.println("");
        return indexSample;
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
    public void svmRandom() {

        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "Try1HandFull \n");
        List<Integer> index = getRandomIndex(1000);
        Mat trainingDataMat = getDataSVMEdgeDELETE("Try1HandFull", index, true);
        int rows = trainingDataMat.rows();
        Mat labelsMat = getLabel(trainingDataMat.rows(), 12345);
        Mat sampleDataMat = getDataSVMEdgeDELETE("Try1HandFull", index, false);
        List<String> fileName = getFileName("Try1HandFull", index, false);
        List<String> fileNameT = getFileName("Try1HandFull", index, true);

        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "Try1JempolKelingking \n");
        trainingDataMat.push_back(getDataSVMEdgeDELETE("Try1JempolKelingking", index, true));
        labelsMat.push_back(getLabel(rows, 15));
        sampleDataMat.push_back(getDataSVMEdgeDELETE("Try1JempolKelingking", index, false));
        fileName.addAll(getFileName("Try1JempolKelingking", index, false));
        fileNameT.addAll(getFileName("Try1JempolKelingking", index, true));

        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "Try1JempolTelunjuk \n");
        trainingDataMat.push_back(getDataSVMEdgeDELETE("Try1JempolTelunjuk", index, true));
        labelsMat.push_back(getLabel(rows, 12));
        sampleDataMat.push_back(getDataSVMEdgeDELETE("Try1JempolTelunjuk", index, false));
        fileName.addAll(getFileName("Try1JempolTelunjuk", index, false));
        fileNameT.addAll(getFileName("Try1JempolTelunjuk", index, true));

        //######################################################################
        System.gc();
        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "Try1JempolTelunjukKelingking \n");
        trainingDataMat.push_back(getDataSVMEdgeDELETE("Try1JempolTelunjukKelingking", index, true));
        labelsMat.push_back(getLabel(rows, 125));
        sampleDataMat.push_back(getDataSVMEdgeDELETE("Try1JempolTelunjukKelingking", index, false));
        fileName.addAll(getFileName("Try1JempolTelunjukKelingking", index, false));
        fileNameT.addAll(getFileName("Try1JempolTelunjukKelingking", index, true));

        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "Try1JempolTelunjukTengah \n");
        trainingDataMat.push_back(getDataSVMEdgeDELETE("Try1JempolTelunjukTengah", index, true));
        labelsMat.push_back(getLabel(rows, 123));
        sampleDataMat.push_back(getDataSVMEdgeDELETE("Try1JempolTelunjukTengah", index, false));
        fileName.addAll(getFileName("Try1JempolTelunjukTengah", index, false));
        fileNameT.addAll(getFileName("Try1JempolTelunjukTengah", index, true));

        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "Try1telunjuk \n\n");
        trainingDataMat.push_back(getDataSVMEdgeDELETE("Try1telunjuk", index, true));
        labelsMat.push_back(getLabel(rows, 2));
        sampleDataMat.push_back(getDataSVMEdgeDELETE("Try1telunjuk", index, false));
        fileName.addAll(getFileName("Try1telunjuk", index, false));
        fileNameT.addAll(getFileName("Try1telunjuk", index, true));

        //######################################################################
        System.gc();
        //######################################################################
        svm.train(trainingDataMat, Ml.ROW_SAMPLE, labelsMat);
        svm.save("E:\\TA\\hCoba.xml");
//        trainingDataMat = null;
        labelsMat = null;
        System.gc();
        //######################################################################

        int[][] confusionMatrix = new int[6][6];
        txtAreaStatus.setText(txtAreaStatus.getText() + "sampleDataMat.rows() " + sampleDataMat.rows() + " \n");
        for (int j = 0; j < sampleDataMat.rows(); j++) {
            float label = svm.predict(sampleDataMat.row(j));
            int l = 0, m = 0;
            if (fileName.get(j).contains("HandFull")) {
                l = 0;
            } else if (fileName.get(j).contains("JempolKelingking")) {
                l = 1;
            } else if (fileName.get(j).contains("JempolTelunjukKelingking")) {
                l = 2;
            } else if (fileName.get(j).contains("JempolTelunjukTengah")) {
                l = 3;
            } else if (fileName.get(j).contains("JempolTelunjuk")) {
                l = 4;
            } else if (fileName.get(j).contains("telunjuk")) {
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
//            txtAreaStatus.setText(txtAreaStatus.getText() + fileName.get(j) + " : " + label + " \n");

        }
        txtAreaStatus.setText(txtAreaStatus.getText() + " \n");
        for (int i = 0; i < 6; i++) {
            for (int k = 0; k < 6; k++) {
                txtAreaStatus.setText(txtAreaStatus.getText() + confusionMatrix[i][k] + " ");
            }
            txtAreaStatus.setText(txtAreaStatus.getText() + " \n");
        }
        confusionMatriks(confusionMatrix, sampleDataMat.rows(), false);
        svm.save("E:\\TA\\hCoba.xml");
        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + " \n");
        confusionMatrix = new int[6][6];
        txtAreaStatus.setText(txtAreaStatus.getText() + "trainingDataMat.rows() " + trainingDataMat.rows() + " \n"
        );
        System.out.println("fileNameT.get(j)" + fileNameT.size());
        for (int j = 0; j < trainingDataMat.rows(); j++) {
            float label = svm.predict(trainingDataMat.row(j));
            int l = 0, m = 0;
            if (fileNameT.get(j).contains("HandFull")) {
                l = 0;
            } else if (fileNameT.get(j).contains("JempolKelingking")) {
                l = 1;
            } else if (fileNameT.get(j).contains("JempolTelunjukKelingking")) {
                l = 2;
            } else if (fileNameT.get(j).contains("JempolTelunjukTengah")) {
                l = 3;
            } else if (fileNameT.get(j).contains("JempolTelunjuk")) {
                l = 4;
            } else if (fileNameT.get(j).contains("telunjuk")) {
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
//            txtAreaStatus.setText(txtAreaStatus.getText() + fileNameT.get(j) + " : " + label + " \n");
        }
        for (int i = 0; i < 6; i++) {
            for (int k = 0; k < 6; k++) {
                txtAreaStatus.setText(txtAreaStatus.getText() + confusionMatrix[i][k] + " ");
            }
            txtAreaStatus.setText(txtAreaStatus.getText() + " \n");
        }
        confusionMatriks(confusionMatrix, trainingDataMat.rows(), true);
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
    public void svmBisindoRandom() {

        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "I_used \n");
        List<Integer> index = getRandomIndex(1200);
        Mat trainingDataMat = getDataSVMEdgeDELETE("I_used", index, true);
        int rows = trainingDataMat.rows();
        Mat labelsMat = getLabel(trainingDataMat.rows(), 12345);
        Mat sampleDataMat = getDataSVMEdgeDELETE("I_used", index, false);
        List<String> fileName = getFileName("I_used", index, false);
        List<String> fileNameT = getFileName("I_used", index, true);

        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "L_used \n");
        trainingDataMat.push_back(getDataSVMEdgeDELETE("L_used", index, true));
        labelsMat.push_back(getLabel(rows, 15));
        sampleDataMat.push_back(getDataSVMEdgeDELETE("L_used", index, false));
        fileName.addAll(getFileName("L_used", index, false));
        fileNameT.addAll(getFileName("L_used", index, true));

        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "O_used \n");
        trainingDataMat.push_back(getDataSVMEdgeDELETE("O_used", index, true));
        labelsMat.push_back(getLabel(rows, 12));
        sampleDataMat.push_back(getDataSVMEdgeDELETE("O_used", index, false));
        fileName.addAll(getFileName("O_used", index, false));
        fileNameT.addAll(getFileName("O_used", index, true));

        //######################################################################
        System.gc();
        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "U_used \n");
        trainingDataMat.push_back(getDataSVMEdgeDELETE("U_used", index, true));
        labelsMat.push_back(getLabel(rows, 125));
        sampleDataMat.push_back(getDataSVMEdgeDELETE("U_used", index, false));
        fileName.addAll(getFileName("U_used", index, false));
        fileNameT.addAll(getFileName("U_used", index, true));

        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "V_used \n");
        trainingDataMat.push_back(getDataSVMEdgeDELETE("V_used", index, true));
        labelsMat.push_back(getLabel(rows, 123));
        sampleDataMat.push_back(getDataSVMEdgeDELETE("V_used", index, false));
        fileName.addAll(getFileName("V_used", index, false));
        fileNameT.addAll(getFileName("V_used", index, true));

        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "C_used \n");
        trainingDataMat.push_back(getDataSVMEdgeDELETE("C_used", index, true));
        labelsMat.push_back(getLabel(rows, 2));
        sampleDataMat.push_back(getDataSVMEdgeDELETE("C_used", index, false));
        fileName.addAll(getFileName("C_used", index, false));
        fileNameT.addAll(getFileName("C_used", index, true));

        //######################################################################
        System.gc();
        //######################################################################
        svm.train(trainingDataMat, Ml.ROW_SAMPLE, labelsMat);
        svm.save("E:\\TA\\Bisindo.xml");
//        trainingDataMat = null;
        labelsMat = null;
        System.gc();
        //######################################################################

        int[][] confusionMatrix = new int[6][6];
        txtAreaStatus.setText(txtAreaStatus.getText() + "sampleDataMat.rows() " + sampleDataMat.rows() + " \n");
        for (int j = 0; j < sampleDataMat.rows(); j++) {
            float label = svm.predict(sampleDataMat.row(j));
            int l = 0, m = 0;
            if (fileName.get(j).contains("I")) {
                l = 0;
            } else if (fileName.get(j).contains("L")) {
                l = 1;
            } else if (fileName.get(j).contains("O")) {
                l = 2;
            } else if (fileName.get(j).contains("U")) {
                l = 3;
            } else if (fileName.get(j).contains("V")) {
                l = 4;
            } else if (fileName.get(j).contains("C")) {
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
//            txtAreaStatus.setText(txtAreaStatus.getText() + fileName.get(j) + " : " + label + " \n");

        }
        txtAreaStatus.setText(txtAreaStatus.getText() + " \n");
        for (int i = 0; i < 6; i++) {
            for (int k = 0; k < 6; k++) {
                txtAreaStatus.setText(txtAreaStatus.getText() + confusionMatrix[i][k] + " ");
            }
            txtAreaStatus.setText(txtAreaStatus.getText() + " \n");
        }
        confusionMatriks(confusionMatrix, sampleDataMat.rows(), false);

        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + " \n");
        confusionMatrix = new int[6][6];
        txtAreaStatus.setText(txtAreaStatus.getText() + "trainingDataMat.rows() " + trainingDataMat.rows() + " \n");
        for (int j = 0; j < trainingDataMat.rows(); j++) {
            float label = svm.predict(trainingDataMat.row(j));
            int l = 0, m = 0;
            if (fileNameT.get(j).contains("I")) {
                l = 0;
            } else if (fileNameT.get(j).contains("L")) {
                l = 1;
            } else if (fileNameT.get(j).contains("O")) {
                l = 2;
            } else if (fileNameT.get(j).contains("U")) {
                l = 3;
            } else if (fileNameT.get(j).contains("V")) {
                l = 4;
            } else if (fileNameT.get(j).contains("C")) {
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
//            txtAreaStatus.setText(txtAreaStatus.getText() + fileNameT.get(j) + " : " + label + " \n");

        }
        for (int i = 0; i < 6; i++) {
            for (int k = 0; k < 6; k++) {
                txtAreaStatus.setText(txtAreaStatus.getText() + confusionMatrix[i][k] + " ");
            }
            txtAreaStatus.setText(txtAreaStatus.getText() + " \n");
        }
        confusionMatriks(confusionMatrix, trainingDataMat.rows(), true);
        svm.save("E:\\TA\\Bisindo.xml");
    }
//######################################################################

    /**
     * method untuk menghitung akurasi
     * var:
     * float TP, TN, FP, FN: fariabel True, Positif, False, Negatif
     */
    public void confusionMatriks(int[][] predict, int jumlahData, boolean train) {
        float TP = 0, TN = 0, FP = 0, FN = 0;
        for (int i = 0; i < predict.length; i++) {
            for (int j = 0; j < predict.length; j++) {
                for (int k = 0; k < predict.length; k++) {
                    if (i != j && i != k) {
                        TN += predict[j][k];
                    }
                }
                if (i == j) {
                    TP += predict[i][j];
                } else {
                    FN += predict[i][j];
                    FP += predict[j][i];
                }
            }

        }
        float precision, recall, accuracy;
        precision = TP / (TP + FP);
        recall = TP / (TP + FN);
        accuracy = TP / jumlahData;
        txtAreaStatus.setText(txtAreaStatus.getText() + "TP: " + TP + " \n");
        txtAreaStatus.setText(txtAreaStatus.getText() + "TN: " + TN + " \n");
        txtAreaStatus.setText(txtAreaStatus.getText() + "FP: " + FP + " \n");
        txtAreaStatus.setText(txtAreaStatus.getText() + "FN: " + FN + " \n");
        txtAreaStatus.setText(txtAreaStatus.getText() + "precision: " + precision + " \n");
        txtAreaStatus.setText(txtAreaStatus.getText() + "recall: " + recall + " \n");
        txtAreaStatus.setText(txtAreaStatus.getText() + "accuracy: " + accuracy + " \n\n");
        if (train) {
            akurasiSeedTrain.add(Double.valueOf(accuracy));
        } else {
            akurasiSeedSample.add(Double.valueOf(accuracy));
        }
    }
    //######################################################################

    /**
     * method untuk menghitung rata-rata akurasi
     * var:
     * float i : menghitung jumlah akurasi
     */
    private void rataRataAkurasiSeed() {
        float i = 0;
        for (Double integer : akurasiSeedTrain) {
            i += integer;
        }
        i /= akurasiSeedTrain.size();
        txtAreaStatus.setText(txtAreaStatus.getText() + "Rata-rata akurasi Train: " + i + " \n\n");

        i = 0;
        for (Double integer : akurasiSeedSample) {
            i += integer;
        }
        i /= akurasiSeedSample.size();
        txtAreaStatus.setText(txtAreaStatus.getText() + "Rata-rata akurasi Sample: " + i + " \n\n");
    }
    //######################################################################
    //######################################################################
    //######################################################################

    /**
     * method untuk menghitung rata-rata akurasi
     * var:
     * float i : menghitung jumlah akurasi
     */
    public void svmHogRandom() {

        //######################################################################
        System.out.println("time : " + java.time.LocalTime.now().compareTo(time));
        txtAreaStatus.setText(txtAreaStatus.getText() + "Try1HandFull \n");
        List<Integer> index = getRandomIndex(1000);
        Mat trainingDataMat = getDataSVMHogDELETE("Try1HandFull", index, true);
        int rows = trainingDataMat.rows();
        Mat labelsMat = getLabel(trainingDataMat.rows(), 12345);
        Mat sampleDataMat = getDataSVMHogDELETE("Try1HandFull", index, false);
        List<String> fileName = getFileName("Try1HandFull", index, false);
        List<String> fileNameT = getFileName("Try1HandFull", index, true);

        System.out.println(java.time.LocalTime.now() + " Try1HandFull : " + java.time.LocalTime.now().compareTo(time));
        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "Try1JempolKelingking \n");
        trainingDataMat.push_back(getDataSVMHogDELETE("Try1JempolKelingking", index, true));
        labelsMat.push_back(getLabel(rows, 15));
        sampleDataMat.push_back(getDataSVMHogDELETE("Try1JempolKelingking", index, false));
        fileName.addAll(getFileName("Try1JempolKelingking", index, false));
        fileNameT.addAll(getFileName("Try1JempolKelingking", index, true));

        System.out.println(java.time.LocalTime.now() + " Try1JempolKelingking : " + java.time.LocalTime.now().compareTo(time));
        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "Try1JempolTelunjuk \n");
        trainingDataMat.push_back(getDataSVMHogDELETE("Try1JempolTelunjuk", index, true));
        labelsMat.push_back(getLabel(rows, 12));
        sampleDataMat.push_back(getDataSVMHogDELETE("Try1JempolTelunjuk", index, false));
        fileName.addAll(getFileName("Try1JempolTelunjuk", index, false));
        fileNameT.addAll(getFileName("Try1JempolTelunjuk", index, true));

        System.out.println(java.time.LocalTime.now() + " Try1JempolTelunjuk : " + java.time.LocalTime.now().compareTo(time));
        //######################################################################
        System.gc();
        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "Try1JempolTelunjukKelingking \n");
        trainingDataMat.push_back(getDataSVMHogDELETE("Try1JempolTelunjukKelingking", index, true));
        labelsMat.push_back(getLabel(rows, 125));
        sampleDataMat.push_back(getDataSVMHogDELETE("Try1JempolTelunjukKelingking", index, false));
        fileName.addAll(getFileName("Try1JempolTelunjukKelingking", index, false));
        fileNameT.addAll(getFileName("Try1JempolTelunjukKelingking", index, true));

        System.out.println(java.time.LocalTime.now() + " Try1JempolTelunjukKelingking : " + java.time.LocalTime.now().compareTo(time));
        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "Try1JempolTelunjukTengah \n");
        trainingDataMat.push_back(getDataSVMHogDELETE("Try1JempolTelunjukTengah", index, true));
        labelsMat.push_back(getLabel(rows, 123));
        sampleDataMat.push_back(getDataSVMHogDELETE("Try1JempolTelunjukTengah", index, false));
        fileName.addAll(getFileName("Try1JempolTelunjukTengah", index, false));
        fileNameT.addAll(getFileName("Try1JempolTelunjukTengah", index, true));

        System.out.println(java.time.LocalTime.now() + " Try1JempolTelunjukTengah : " + java.time.LocalTime.now().compareTo(time));
        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + "Try1telunjuk \n\n");
        trainingDataMat.push_back(getDataSVMHogDELETE("Try1telunjuk", index, true));
        labelsMat.push_back(getLabel(rows, 2));
        sampleDataMat.push_back(getDataSVMHogDELETE("Try1telunjuk", index, false));
        fileName.addAll(getFileName("Try1telunjuk", index, false));
        fileNameT.addAll(getFileName("Try1telunjuk", index, true));

        System.out.println(java.time.LocalTime.now() + " Try1telunjuk : " + java.time.LocalTime.now().compareTo(time));
        //######################################################################
        System.gc();
        //######################################################################
        svm.train(trainingDataMat, Ml.ROW_SAMPLE, labelsMat);

        System.out.println(java.time.LocalTime.now() + " train : " + java.time.LocalTime.now().compareTo(time));
        svm.save("E:\\TA\\hCoba.xml");
//        trainingDataMat = null;
        labelsMat = null;
        System.gc();
        //######################################################################

        int[][] confusionMatrix = new int[6][6];
        txtAreaStatus.setText(txtAreaStatus.getText() + "sampleDataMat.rows() " + sampleDataMat.rows() + " \n");
        for (int j = 0; j < sampleDataMat.rows(); j++) {
            float label = svm.predict(sampleDataMat.row(j));
            int l = 0, m = 0;
            if (fileName.get(j).contains("HandFull")) {
                l = 0;
            } else if (fileName.get(j).contains("JempolKelingking")) {
                l = 1;
            } else if (fileName.get(j).contains("JempolTelunjukKelingking")) {
                l = 2;
            } else if (fileName.get(j).contains("JempolTelunjukTengah")) {
                l = 3;
            } else if (fileName.get(j).contains("JempolTelunjuk")) {
                l = 4;
            } else if (fileName.get(j).contains("telunjuk")) {
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
//            txtAreaStatus.setText(txtAreaStatus.getText() + fileName.get(j) + " : " + label + " \n");

        }

        System.out.println(java.time.LocalTime.now() + " testS : " + java.time.LocalTime.now().compareTo(time));
        txtAreaStatus.setText(txtAreaStatus.getText() + " \n");
        for (int i = 0; i < 6; i++) {
            for (int k = 0; k < 6; k++) {
                txtAreaStatus.setText(txtAreaStatus.getText() + confusionMatrix[i][k] + " ");
            }
            txtAreaStatus.setText(txtAreaStatus.getText() + " \n");
        }
        confusionMatriks(confusionMatrix, sampleDataMat.rows(), false);
        svm.save("E:\\TA\\hCoba.xml");
        //######################################################################
        txtAreaStatus.setText(txtAreaStatus.getText() + " \n");
        confusionMatrix = new int[6][6];
        txtAreaStatus.setText(txtAreaStatus.getText() + "trainingDataMat.rows() " + trainingDataMat.rows() + " \n"
        );
        System.out.println("fileNameT.get(j)" + fileNameT.size());
        for (int j = 0; j < trainingDataMat.rows(); j++) {
            float label = svm.predict(trainingDataMat.row(j));
            int l = 0, m = 0;
            if (fileNameT.get(j).contains("HandFull")) {
                l = 0;
            } else if (fileNameT.get(j).contains("JempolKelingking")) {
                l = 1;
            } else if (fileNameT.get(j).contains("JempolTelunjukKelingking")) {
                l = 2;
            } else if (fileNameT.get(j).contains("JempolTelunjukTengah")) {
                l = 3;
            } else if (fileNameT.get(j).contains("JempolTelunjuk")) {
                l = 4;
            } else if (fileNameT.get(j).contains("telunjuk")) {
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
//            txtAreaStatus.setText(txtAreaStatus.getText() + fileNameT.get(j) + " : " + label + " \n");
        }

        System.out.println(java.time.LocalTime.now() + " testT : " + java.time.LocalTime.now().compareTo(time));
        for (int i = 0; i < 6; i++) {
            for (int k = 0; k < 6; k++) {
                txtAreaStatus.setText(txtAreaStatus.getText() + confusionMatrix[i][k] + " ");
            }
            txtAreaStatus.setText(txtAreaStatus.getText() + " \n");
        }
        confusionMatriks(confusionMatrix, trainingDataMat.rows(), true);
        svm.save("E:\\TA\\hCoba.xml");
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
    public Mat getDataSVMHogDELETE(String lokasi, List<Integer> index, Boolean train) {

        File folder;
        if (cmbType.getValue().contains("Bisindo")) {
            folder = new File("E:\\TA\\HandLearnSVM\\BISINDO\\" + lokasi);
        } else {
            folder = new File("E:\\TA\\HandLearnSVM\\Try1\\" + lokasi);
        }
        File[] listOfFiles = folder.listFiles();
        Mat trainingDataMat;
        if (train) {
            trainingDataMat = new Mat(listOfFiles.length - index.size(), 192780, CvType.CV_32FC1);
//            trainingDataMat = new Mat();
        } else {
            trainingDataMat = new Mat(index.size(), 192780, CvType.CV_32FC1);
//            trainingDataMat = new Mat();

        }

        int row = 0;
        HOGDescriptor gDescriptor = new HOGDescriptor();
        for (int i = 0; i < listOfFiles.length; i++) {
            if (!index.contains(i) && train) {
                Mat hand = Imgcodecs.imread(folder.getPath() + "\\" + listOfFiles[i].getName(), CvType.CV_32F);
                Imgproc.resize(hand, hand, new Size(192, 144));
                MatOfFloat descriptors = new MatOfFloat();
                gDescriptor.compute(hand, descriptors);
                float[] trainingData = descriptors.toArray();
                for (int j = 0; j < trainingData.length; j++) {
                    trainingData[j] = Math.round(trainingData[j] * 100000) / 100;

                }
                trainingDataMat.put(row, 0, trainingData);
//                System.out.println("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq");
//                System.out.println("");
//                for (int j = 0; j < trainingDataMat.cols(); j += 2000) {
//                    if (j < trainingDataMat.cols() - 1) {
//                        System.out.println("trainingDataMat.get(" + i + ", " + j + ") " + trainingDataMat.get(i, j)[0]);
//                    }
//                }
                row++;
            } else if (index.contains(i) && !train) {
                Mat hand = Imgcodecs.imread(folder.getPath() + "\\" + listOfFiles[i].getName(), CvType.CV_32F);
                Imgproc.resize(hand, hand, new Size(192, 144));
                MatOfFloat descriptors = new MatOfFloat();
                gDescriptor.compute(hand, descriptors);
                float[] trainingData = descriptors.toArray();
                for (int j = 0; j < trainingData.length; j++) {
                    trainingData[j] = Math.round(trainingData[j] * 100000) / 100;
                }
                trainingDataMat.put(row, 0, trainingData);
                row++;
            }

        }
//        System.out.println("PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP");
//        for (int j = 0; j < trainingDataMat.rows(); j++) {
//            for (int k = 0; k < trainingDataMat.cols(); k++) {
//                System.out.println("trainingDataMat.get(" + j + ", " + k + ").length :" + trainingDataMat.get(j, k)[0]);
//            }
//            System.out.println("");
//        }
        return trainingDataMat;
    }
}
