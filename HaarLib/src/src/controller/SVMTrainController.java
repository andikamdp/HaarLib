/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package src.controller;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringReader;
import java.net.URL;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.ResourceBundle;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.layout.AnchorPane;
import javafx.stage.DirectoryChooser;
import org.opencv.core.Mat;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import src.entity.Data;
import src.utils.DataTrainingPrep;

/**
 * FXML Controller class
 *
 * @author Andika Mulyawan
 */
public class SVMTrainController implements Initializable {

    @FXML
    private Button btnTrain;
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
    @FXML
    private Button btnBrowsClassification;
    @FXML
    private Label lblClassificationLocation;
    @FXML
    private Button btnSaveClasification;
    @FXML
    private TextField txtBoxFileName;
//
    private int ratio;
    private MainAppController mainAppController;
    private SVM svm;
    private int seed;
    private List<Double> akurasiSeedSample;
    private List<Double> akurasiSeedTrain;
//

    /**
     * Initializes the controller class.
     *
     * @param url
     * @param rb
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
        akurasiSeedTrain = new ArrayList<>();
        akurasiSeedSample = new ArrayList<>();
    }

    /**
     * ######################################################################
     * OnClick Action memulai Training diikuti Testing
     * training dan testing akan dilakukan sebanyak 10 kali dengan data teracak setiap training
     * training dengan deskripsi gambar edge dan hog
     * var:
     *
     */
    @FXML
    private void trainOnClick(ActionEvent event) {
        akurasiSeedTrain.clear();
        akurasiSeedSample.clear();
        if (cmbType.getValue().equals("Edge")) {
            txtAreaStatus.setText(txtAreaStatus.getText() + "Train SVM Sampel Acak \n");
            txtAreaStatus.setText(txtAreaStatus.getText() + lblImageLocation.getText() + " \n");
            txtAreaStatus.setText(txtAreaStatus.getText() + "Waktu Mulai Program " + LocalTime.now() + " \n");
            for (int i = 0; i < 20; i++) {
                txtAreaStatus.setText(txtAreaStatus.getText() + "Train SVM Iterasi " + (i + 1) + " \n");
                seed = i;
                svmEdgeRandom();
            }
            rataRataAkurasiSeed();
        } else if (cmbType.getValue().equals("Hog")) {
            txtAreaStatus.setText(txtAreaStatus.getText() + "Train SVM Hog Sampel Acak \n");
            txtAreaStatus.setText(txtAreaStatus.getText() + "Waktu Mulai Program " + LocalTime.now() + " \n");
//            for (int i = 0; i < 10; i++) {
//                txtAreaStatus.setText(txtAreaStatus.getText() + "Train SVM Iterasi" + (i + 1) + " \n");
            seed = 0;
            svmHogRandom();
//            }
            rataRataAkurasiSeed();
        }

    }

    /**
     * ######################################################################
     * method set Main Controller
     *
     */
    void setMainController(MainAppController aThis) {
        mainAppController = aThis;
    }

    /**
     * ######################################################################
     * method untuk train SVM dengan deskripsi gambar Edge
     * var:
     * SVM svm : variabel klasifikasi dengan SVM
     * Mat trainingDataMat : variabel penampung data training
     * Mat labelsMat : variabel penampung label data training
     * Mat sampleDataMat : variabel penampung data testing
     * int rows :
     * File file : variabel penampung lokasi file data gambar
     * File[] listOfFiles : variabe penampung isi file data gambar
     * int[][] confusionMatrix : variabel penampung confussion matriks
     */
    public void svmEdgeRandom() {

        File file = new File(lblImageLocation.getText());
        File[] listFiles = file.listFiles();
//######################################################################
        List<Integer> index = getRandomIndex(listFiles[0].listFiles().length);
        List<String> labels = new ArrayList<>();
        //
        File files;
        int rows = 0;
        Mat dataTraining = new Mat();
        List<Data> trainingDataMat = new ArrayList<>();
        List<Data> sampleDataMat = new ArrayList<>();
        Mat labelsMat = new Mat();
        List<String> fileNameT = new ArrayList<>();
        List<String> fileName = new ArrayList<>();
        //
        System.out.println("Data Prepare" + LocalTime.now());
        for (int i = 0; i < listFiles.length; i++) {
            files = listFiles[i];
            trainingDataMat.addAll(DataTrainingPrep.getDataSVMEdge(files.getAbsolutePath(), index, true));
            sampleDataMat.addAll(DataTrainingPrep.getDataSVMEdge(files.getAbsolutePath(), index, false));
            if (i == 0) {
                rows = trainingDataMat.size();
            }
            labelsMat.push_back(DataTrainingPrep.getLabel(rows, i));
            labels.add(files.getName());
            txtAreaStatus.setText(txtAreaStatus.getText() + files.getName() + " \n");
        }
        txtAreaStatus.setText(txtAreaStatus.getText() + "Waktu Selesai Menyiapkan Data " + LocalTime.now() + " \n");

        dataTraining = DataTrainingPrep.getDataMat(trainingDataMat);
        svm.train(dataTraining, Ml.ROW_SAMPLE, labelsMat);
        txtAreaStatus.setText(txtAreaStatus.getText() + "Waktu Selesai Training " + LocalTime.now() + " \n");
        //######################################################################
        int[][] confusionMatrix = predictClassifier(labels, sampleDataMat);
        confusionMatriks(confusionMatrix, sampleDataMat.size(), false);
        txtAreaStatus.setText(txtAreaStatus.getText() + "Waktu Selesai Testing Data Testing " + LocalTime.now() + " \n");
        //######################################################################
        confusionMatrix = predictClassifier(labels, trainingDataMat);
        confusionMatriks(confusionMatrix, trainingDataMat.size(), true);
        txtAreaStatus.setText(txtAreaStatus.getText() + "Waktu Selesai Testing Data Training " + LocalTime.now() + " \n");
        System.gc();
    }

    /**
     * ######################################################################
     * method untuk train SVM dengan deskripsi gambar Hog
     * var:
     * SVM svm : variabel klasifikasi dengan SVM
     * Mat trainingDataMat : variabel penampung data training
     * Mat labelsMat : variabel penampung label data training
     * Mat sampleDataMat : variabel penampung data testing
     * int rows :
     * File file : variabel penampung lokasi file data gambar
     * File[] listOfFiles : variabe penampung isi file data gambar
     * int[][] confusionMatrix : variabel penampung confussion matriks
     */
    public void svmHogRandom() {
        File file = new File(lblImageLocation.getText());
        File[] listFiles = file.listFiles();
//######################################################################
        List<Integer> index = getRandomIndex(listFiles[0].listFiles().length);
        List<String> labels = new ArrayList<>();
        //
        File files;
        int rows = 0;
        Mat dataTraining = new Mat();
        List<Data> trainingDataMat = new ArrayList<>();
        List<Data> sampleDataMat = new ArrayList<>();
        Mat labelsMat = new Mat();
        for (int i = 0; i < listFiles.length; i++) {
            //
            files = listFiles[i];
            trainingDataMat.addAll(DataTrainingPrep.getDataSVMHog(files.getAbsolutePath(), index, true));
            sampleDataMat.addAll(DataTrainingPrep.getDataSVMHog(files.getAbsolutePath(), index, false));
            if (i == 0) {
                rows = trainingDataMat.size();
            }
            labelsMat.push_back(DataTrainingPrep.getLabel(rows, i));
            labels.add(files.getName());
            txtAreaStatus.setText(txtAreaStatus.getText() + files.getName() + " \n");

        }
        txtAreaStatus.setText(txtAreaStatus.getText() + "Waktu Selesai Menyiapkan Data " + LocalTime.now() + " \n");
        dataTraining = DataTrainingPrep.getDataMat(trainingDataMat);
        svm.train(dataTraining, Ml.ROW_SAMPLE, labelsMat);
        txtAreaStatus.setText(txtAreaStatus.getText() + "Waktu Selesai Training " + LocalTime.now() + " \n");
        //######################################################################
        int[][] confusionMatrix = predictClassifier(labels, sampleDataMat);
        confusionMatriks(confusionMatrix, sampleDataMat.size(), false);
        txtAreaStatus.setText(txtAreaStatus.getText() + "Waktu Selesai Testing Data Testing " + LocalTime.now() + " \n");
        //######################################################################
        confusionMatrix = predictClassifier(labels, trainingDataMat);
        confusionMatriks(confusionMatrix, trainingDataMat.size(), true);
        txtAreaStatus.setText(txtAreaStatus.getText() + "Waktu Selesai Testing Data Training " + LocalTime.now() + " \n");
        System.gc();
    }

    /**
     * ######################################################################
     * OnClick Action untuk memperoleh lokasi data gambar
     * var:
     * DirectoryChooser brows :
     * File Path :
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
     * method untuk menghitung akurasi training
     * var:
     * float TP, TN, FP, FN : fariabel penampung nilai True, Positif, False, Negatif
     * float precision, recall, accuracy :
     *
     * @param predict
     * @param jumlahData
     * @param train
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
     * param:
     * int jumlahData : variabel penampung jumlah data training perClass(label)
     * var:
     * List<Integer> index : variabel penampung urutan angka sebanyak jumlahData
     * List<Integer> indexSample : variabel penampung angka random
     * Random rand : variabel penampung jumlah data training
     * int numberOfElements :
     *
     *
     * @param jumlahData
     * @return
     */
    public List<Integer> getRandomIndex(int jumlahData) {
        List<Integer> index = new ArrayList<>();
        List<Integer> indexSample = new ArrayList<>();
        for (int i = 0; i < jumlahData; i++) {
            index.add(i);
        }
        Random rand = new Random(seed);
        int numberOfElements = (int) (((double) ratio / 100.0) * (double) jumlahData);
        for (int i = 0; i < numberOfElements; i++) {
            int randomIndex = rand.nextInt(index.size());
            indexSample.add(index.get(randomIndex));
            index.remove(randomIndex);
        }
        Collections.sort(indexSample);
        for (Integer integer : indexSample) {
        }
        return indexSample;
    }

    /**
     * ######################################################################
     * method untuk
     * param:
     *
     * var:
     *
     * @param fileName
     * @param labels
     * @param sampleDataMat
     * @return
     */
    public int[][] predictClassifier(List<String> labels, List<Data> sampleDataMat) {
        int[][] confusionMatrix = new int[6][6];
        for (int j = 0; j < sampleDataMat.size(); j++) {
            float label = svm.predict(sampleDataMat.get(j).getDataMat());
            int l = 0, m = 0;
            if (sampleDataMat.get(j).getDataName().contains(labels.get(0).substring(0, 1))) {
                l = 0;
            } else if (sampleDataMat.get(j).getDataName().contains(labels.get(1).substring(0, 1))) {
                l = 1;
            } else if (sampleDataMat.get(j).getDataName().contains(labels.get(2).substring(0, 1))) {
                l = 2;
            } else if (sampleDataMat.get(j).getDataName().contains(labels.get(3).substring(0, 1))) {
                l = 3;
            } else if (sampleDataMat.get(j).getDataName().contains(labels.get(4).substring(0, 1))) {
                l = 4;
            } else if (sampleDataMat.get(j).getDataName().contains(labels.get(5).substring(0, 1))) {
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
        txtAreaStatus.setText(txtAreaStatus.getText() + " \n");
        return confusionMatrix;
    }

    @FXML
    private void browsClassificationOnClick(ActionEvent event) {
        DirectoryChooser brows = new DirectoryChooser();
        brows.setTitle("Buka Folder Classification Save Lokasi");
        File Path = brows.showDialog(apTrainWindow.getScene().getWindow());
        if (Path != null) {
            lblClassificationLocation.setText(Path.getAbsolutePath());
        }
    }

    @FXML
    private void saveClasssificationOnClilck(ActionEvent event) {
        try {
            String string = txtAreaStatus.getText();
            File file = new File(lblClassificationLocation.getText() + "\\" + txtBoxFileName.getText() + ".txt");
            try (
                    BufferedReader reader = new BufferedReader(new StringReader(string));
                    PrintWriter writer = new PrintWriter(new FileWriter(file));) {
                reader.lines().forEach(new Consumer<String>() {
                    @Override
                    public void accept(String line) {
                        writer.println(line);
                    }
                });
            }
        } catch (IOException ex) {
            Logger.getLogger(SVMTrainController.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
