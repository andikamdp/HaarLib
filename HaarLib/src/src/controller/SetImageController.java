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
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import src.Utils;
import src.utils.Preprocessing;

/**
 * FXML Controller class
 *
 * @author Andika Mulyawan
 */
public class SetImageController implements Initializable {

    @FXML
    private ImageView layarBW;
    @FXML
    private ImageView layarEdge;
    @FXML
    private Button btnflipImage;
    @FXML
    private Button btnNextImage;
    @FXML
    private Button btnSaveImage;
    @FXML
    private TextField txtBoxFileName;
    @FXML
    private TextField txtBoxFolderName;
    @FXML
    private TextField txtS;
    @FXML
    private TextField txtValue;
    @FXML
    private ImageView layarMain;
    private String location;
    @FXML
    private Button btnflipImage1;
    @FXML
    private Button btnOpenImage;
///
    private File folder;
    private File[] listOfFiles;
    private int indexFile, index;

    /**
     * Initializes the controller class.
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
        location = "";
    }

    @FXML
    private void switchImageBWToMain(MouseEvent event) {

        Image Mn = layarBW.getImage();
        layarBW.setImage(layarMain.getImage());
        layarMain.setImage(Mn);
    }

    @FXML
    private void switchImageEdgeToMain(MouseEvent event) {

        Image Mn = layarEdge.getImage();
        layarEdge.setImage(layarMain.getImage());
        layarMain.setImage(Mn);
    }

    @FXML
    private void FlipImageOnClick(ActionEvent event) {

    }

    @FXML
    private void showNextImageOnClick(ActionEvent event) {

        Mat hand = Imgcodecs.imread(folder.getAbsolutePath() + "\\" + listOfFiles[index].getName());
        txtBoxFileName.setText(listOfFiles[index].getName());
//        layarMain.setImage(Utils.mat2Image(hand));
//        Mat tresholdedHand = Preprocessing.segment(hand.clone());
//        layarBW.setImage(Utils.mat2Image(tresholdedHand));
//        Start(hand);
        hog(hand);
        if (index < indexFile) {
            index++;
        } else {
            index = 0;
        }
    }

    @FXML
    private void saveImageOnClick(ActionEvent event) {
    }

    @FXML
    private void getFramePointOnClick(MouseEvent event) {
        txtValue.setText(String.valueOf(event.getY()));
        txtS.setText(String.valueOf(event.getX()));
    }
    private MainAppController mainAppController;

    void setMainController(MainAppController aThis) {
        mainAppController = aThis;
    }

    @FXML
    private void openImageOnClick(ActionEvent event) {
        folder = new File(txtBoxFolderName.getText());
        listOfFiles = folder.listFiles();
        indexFile = listOfFiles.length;
        index = 0;
    }

    List<Integer> Puncak = new ArrayList<>();
    List<Integer> Lembah = new ArrayList<>();
    List<Integer> devContourIdxList;

    private void hapusTitik(List<MatOfPoint> contours, Mat hand) {
        try {

            Puncak = new ArrayList<>();
            Point[] point = contours.get(0).toArray();
            Puncak.addAll(devContourIdxList);
            Lembah.addAll(devContourIdxList);
//            System.out.println("isi index puncak awal");
//            for (Integer integer : Puncak) {
//                System.out.println(integer);
//            }
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
            System.out.println("contous " + contours.get(0).toArray().length);
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
            Preprocessing.drawPointColor(contours, hand2, Puncak);
            Preprocessing.drawJumlahJari(hand, Puncak.size());
//            layarBW.setImage(Utils.mat2Image(hand2));
//            drawPointColor(contous, hand, devContourIdxList);
            layarEdge.setImage(Utils.mat2Image(hand2));
        } catch (Exception e) {
            System.out.
                    println("hapusTitik(List<MatOfPoint> contours, Mat hand)");
            System.out.println(e);
            System.out.println("");
        }
    }

    private void Start(Mat frame) {
        Puncak = new ArrayList<>();
        Image imageToMat;
        Core.flip(frame, frame, 1);
        Mat hand = Preprocessing.getBox(frame.clone());
//        Mat hand = frame;
//        Mat tresholded = Preprocessing.segment(hand.clone());
        Mat tresholded = Preprocessing.segmentInvers(hand.clone());
        layarBW.setImage(Utils.mat2Image(tresholded));
        imageToMat = Utils.mat2Image(tresholded);

        List<MatOfPoint> contous = Preprocessing.getContour(tresholded);
        Preprocessing.toListContour(contous.get(0));
        List<MatOfInt4> devOfInt4s = Preprocessing.getDevectIndexPoint(contous);
        devContourIdxList = devOfInt4s.get(0).toList();
//        Preprocessing.toListMatOfPointDevec(contous, devOfInt4s, devContourIdxList);
//                        HandRec(contous, hand);
//                        hitungJarakTitik(devOfPoints);
        hapusTitik(contous, frame);
//                        drawJumlahJari(contous, frame, Puncak);
        layarMain.setImage(Utils.mat2Image(frame));
        imageToMat = Utils.mat2Image(hand);

    }

    private void hog(Mat frame) {
//        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
        System.out.println("frame.rows() " + frame.rows());
        System.out.println("frame.cols() " + frame.cols());
        System.out.println("frame.type() " + frame.type());
        System.out.println("frame.channels() " + frame.channels());
//        frame = Preprocessing.segmentInvers(frame);
//        frame.convertTo(frame, CvType.CV_32F);
        Mat gx = new Mat(), gy = new Mat();
        Imgproc.Sobel(frame, gx, CvType.CV_32F, 1, 0);
        Imgproc.Sobel(frame, gy, CvType.CV_32F, 0, 1);
//
        Mat mag = new Mat(), angel = new Mat();
        Core.cartToPolar(gx, gy, mag, angel, true);

        System.out.println("gx.rows() " + gx.rows());
        System.out.println("gx.cols() " + gx.cols());
        System.out.println("gx.type() " + gx.type());
        System.out.println("gx.channels() " + gx.channels());
        System.out.println("gy.rows() " + gy.rows());
        System.out.println("gy.cols() " + gy.cols());
        System.out.println("gy.type() " + gy.type());
        System.out.println("gy.channels() " + gy.channels());
        System.out.println("mag.rows() " + mag.rows());
        System.out.println("mag.cols() " + mag.cols());
        System.out.println("mag.type() " + mag.type());
        System.out.println("mag.channels() " + mag.channels());
        layarMain.setImage(Utils.mat2Image(mag));
        layarBW.setImage(Utils.mat2Image(gx));
        layarEdge.setImage(Utils.mat2Image(gy));
    }
}
