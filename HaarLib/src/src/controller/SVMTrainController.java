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
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.layout.AnchorPane;
import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;
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
import src.utils.DataTrainingPrep;
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
    @FXML
    private Button btnBrowsImage;
    @FXML
    private Label lblImageLocation;
    @FXML
    private AnchorPane apTrainWindow;
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
        cmbType.setItems(getCmbType());
        svm = SVM.create();
        svm.setKernel(SVM.LINEAR);
        svm.setType(SVM.C_SVC);
        seed = 0;
        akurasiSeedTrain = new ArrayList<Double>();
        akurasiSeedSample = new ArrayList<Double>();
    }

    /**
     * ######################################################################
     * OnClick Action memulai Training dan Testing
     * var:
     * float TP, TN, FP, FN: fariabel True, Positif, False, Negatif
     */
    @FXML
    private void trainOnClick(ActionEvent event) {
        akurasiSeedTrain.clear();
        akurasiSeedSample.clear();
        if (cmbType.getValue().equals("Edge")) {
            txtAreaStatus.setText(txtAreaStatus.getText() + "Train SVM Sampel Acak \n");
            for (int i = 0; i < 10; i++) {
                seed = i;
                svmEdgeRandom();
            }
            rataRataAkurasiSeed();
        } else if (cmbType.getValue().equals("Hog")) {
            txtAreaStatus.setText(txtAreaStatus.getText() + "Train SVM Hog Sampel Acak \n");
//            for (int i = 0; i < 10; i++) {
            time = java.time.LocalTime.now();
//            System.out.println("time : " + time);
//            seed = 0;
            svmHogRandom();
//            }
            rataRataAkurasiSeed();
        }

    }

    @FXML
    private void predictOnAction(ActionEvent event) {
    }

    /**
     * ######################################################################
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

    /**
     * ######################################################################
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
    public void svmEdgeRandom() {
        File file = new File(lblImageLocation.getText());
        File[] listFiles = file.listFiles();
        List<Integer> index = getRandomIndex(listFiles[2].listFiles().length);
        List<String> labels = new ArrayList<String>();
        //
        File files;
        Mat trainingDataMat = new Mat();
        int rows = 0;
        Mat labelsMat = new Mat();
        Mat sampleDataMat = new Mat();
        List<String> fileNameT = new ArrayList<>();
        List<String> fileName = new ArrayList<>();
        //
        for (int i = 0; i < listFiles.length; i++) {
            files = listFiles[i];
            txtAreaStatus.setText(txtAreaStatus.getText() + files.getName() + " \n");
            trainingDataMat.push_back(DataTrainingPrep.getDataSVMEdge(files.getAbsolutePath(), index, true));
            if (i == 0) {
                rows = trainingDataMat.rows();
            }
            System.out.println("rows: " + rows);
            labelsMat.push_back(DataTrainingPrep.getLabel(rows, i));
            sampleDataMat.push_back(DataTrainingPrep.getDataSVMEdge(files.getAbsolutePath(), index, false));
            fileNameT.addAll(DataTrainingPrep.getFileName(files.getAbsolutePath(), index, true));
            fileName.addAll(DataTrainingPrep.getFileName(files.getAbsolutePath(), index, false));
            labels.add(files.getName());
        }
        svm.train(trainingDataMat, Ml.ROW_SAMPLE, labelsMat);
        labelsMat = null;
        System.gc();
        //######################################################################
        int[][] confusionMatrix = predictClassifier(fileName, labels, sampleDataMat);
        confusionMatriks(confusionMatrix, sampleDataMat.rows(), false);
        //######################################################################
        confusionMatrix = predictClassifier(fileNameT, labels, trainingDataMat);
        confusionMatriks(confusionMatrix, trainingDataMat.rows(), true);
        svm.save(file.getAbsolutePath() + "\\Edge.xml");
    }

    /**
     * ######################################################################
     * method untuk menghitung rata-rata akurasi
     * var:
     * float i : menghitung jumlah akurasi
     */
    public void svmHogRandom() {
        File file = new File(lblImageLocation.getText());
        File[] listFiles = file.listFiles();
        List<Integer> index = getRandomIndex(listFiles[2].listFiles().length);
        List<String> labels = new ArrayList<String>();
        //
        File files;
        Mat trainingDataMat = new Mat();
        int rows = 0;
        Mat labelsMat = new Mat();
        Mat sampleDataMat = new Mat();
        List<String> fileNameT = new ArrayList<>();
        List<String> fileName = new ArrayList<>();
        //
        for (int i = 0; i < listFiles.length; i++) {
            files = listFiles[i];
            txtAreaStatus.setText(txtAreaStatus.getText() + files.getName() + " \n");
            trainingDataMat.push_back(DataTrainingPrep.getDataSVMHog(files.getAbsolutePath(), index, true));
            if (i == 0) {
                rows = trainingDataMat.rows();
            }
            System.out.println("rows: " + rows);
            labelsMat.push_back(DataTrainingPrep.getLabel(rows, i));
            sampleDataMat.push_back(DataTrainingPrep.getDataSVMHog(files.getAbsolutePath(), index, false));
            fileNameT.addAll(DataTrainingPrep.getFileName(files.getAbsolutePath(), index, true));
            fileName.addAll(DataTrainingPrep.getFileName(files.getAbsolutePath(), index, false));
            labels.add(files.getName());
        }
        svm.train(trainingDataMat, Ml.ROW_SAMPLE, labelsMat);
        labelsMat = null;
        System.gc();
        //######################################################################
        int[][] confusionMatrix = predictClassifier(fileName, labels, sampleDataMat);
        confusionMatriks(confusionMatrix, sampleDataMat.rows(), false);
        //######################################################################
        confusionMatrix = predictClassifier(fileNameT, labels, trainingDataMat);
        confusionMatriks(confusionMatrix, trainingDataMat.rows(), true);
        svm.save(file.getAbsolutePath() + "\\Hog.xml");
    }

    /**
     * ######################################################################
     * OnClick Action untuk memperoleh lokasi data gambar
     * var:
     * float TP, TN, FP, FN: fariabel True, Positif, False, Negatif
     */
    @FXML
    private void browsImageOnClick(ActionEvent event) {
        DirectoryChooser brows = new DirectoryChooser();
        brows.setTitle("Buka Folder Data Training");

        File Path = brows.showDialog(apTrainWindow.getScene().getWindow());
        if (Path != null) {
            lblImageLocation.setText(Path.getAbsolutePath());
        }

    }

    /**
     * ######################################################################
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

    /**
     * ######################################################################
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

    /**
     * ######################################################################
     * method untuk mengisi ComboBox cmbValue
     * var:
     * ObservableList<String> type :
     *
     * @return
     */
    public ObservableList<String> getCmbType() {
        ObservableList<String> type = FXCollections.observableArrayList();
        type.add("Edge");
        type.add("Hog");
        return type;
    }

    /**
     * ######################################################################
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
        System.out.println("");
        return indexSample;
    }

    /**
     * ######################################################################
     * method untuk
     * param:
     *
     * var:
     *
     */
    public int[][] predictClassifier(List<String> fileName, List<String> labels, Mat sampleDataMat) {
        int[][] confusionMatrix = new int[6][6];
        for (int j = 0; j < sampleDataMat.rows(); j++) {
            float label = svm.predict(sampleDataMat.row(j));
            int l = 0, m = 0;
            if (fileName.get(j).contains(labels.get(0))) {
                l = 0;
            } else if (fileName.get(j).contains(labels.get(1).substring(0, 1))) {
                l = 1;
            } else if (fileName.get(j).contains(labels.get(2).substring(0, 1))) {
                l = 2;
            } else if (fileName.get(j).contains(labels.get(3).substring(0, 1))) {
                l = 3;
            } else if (fileName.get(j).contains(labels.get(4).substring(0, 1))) {
                l = 4;
            } else if (fileName.get(j).contains(labels.get(5).substring(0, 1))) {
                l = 5;
            }
            if (label == 0) {
                m = 0;
            } else if (label == 1) {
                m = 1;
            } else if (label == 2) {
                m = 2;
            } else if (label == 3) {
                m = 3;
            } else if (label == 4) {
                m = 4;
            } else if (label == 5) {
                m = 5;
            }
            confusionMatrix[l][m] += 1;
            // txtAreaStatus.setText(txtAreaStatus.getText() + fileName.get(j) + " : " + label + " \n");
        }
        for (int i = 0; i < 6; i++) {
            for (int k = 0; k < 6; k++) {
                txtAreaStatus.setText(txtAreaStatus.getText() + confusionMatrix[i][k] + " ");
            }
            txtAreaStatus.setText(txtAreaStatus.getText() + " \n");
        }
        return confusionMatrix;
    }
}
