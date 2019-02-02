/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package src.controller;

import java.io.File;
import java.net.URL;
import java.nio.file.Files;
import java.util.ResourceBundle;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.AnchorPane;
import javafx.stage.DirectoryChooser;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import src.Utils;
import src.utils.Preprocessing;

/**
 * FXML Controller class
 *
 * @author Andika Mulyawan
 */
public class BuildImageDatasetController implements Initializable {

    @FXML
    private Button btnStartCamera;
    @FXML
    private Button btnBrowsSaveFile;
    @FXML
    private TextField txtSaveFileLocation;
    @FXML
    private Button btnCreateFolder;
    @FXML
    private TextField txtFolderName;
    @FXML
    private TextField txtIndex;
    @FXML
    private TextField txtIndexLabel;
    @FXML
    private ImageView layarMain;
    @FXML
    private AnchorPane apGetImage;
//
    private ScheduledExecutorService timer;
    private VideoCapture capture;
    private boolean cameraActive;
    private int i;
    private MainAppController mainAppController;
    private File imgDir, imgLblDir;
    private Alert alert;

    /**
     * Initializes the controller class.
     *
     * @param url
     * @param rb
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
        capture = new VideoCapture();
        cameraActive = false;
        i = 0;
        txtIndex.setText(String.valueOf(i));
        imgDir = null;
        imgLblDir = null;
        alert = new Alert(Alert.AlertType.ERROR);
    }

    /**
     * ######################################################################
     * Method awal untuk membuka kamera dan memanggil method
     * var:
     * boolean cameraActive :
     * VideoCapture capture :
     * Runnable frameGrabber :
     * Mat frame :
     * ScheduledExecutorService timer :
     * Button btnStartCamera :
     */
    @FXML
    private void startCameraOnClick(ActionEvent event) {
        i = 0;
        if (!txtFolderName.getText().equals("") && !txtSaveFileLocation.getText().equals("") && imgDir.exists() && imgLblDir.exists()) {
            if (!cameraActive) {
                capture.open(0);
                if (this.capture.isOpened()) {
                    cameraActive = true;
                    Runnable frameGrabber = new Runnable() {
                        @Override
                        public void run() {
                            Mat frame = null;
                            try {
                                frame = grabFrame();
                                start(frame);
                            } catch (Exception e) {
                                System.out.println("startCameraOnClick " + e);
                            }
                        }
                    };

                    this.timer = Executors.newSingleThreadScheduledExecutor();

                    timer.scheduleAtFixedRate(frameGrabber,
                            0, 33,
                            TimeUnit.MILLISECONDS);
                    btnStartCamera.setText(
                            "stop Camera");
                } else {
                    System.err.println("tidak dapat membuka kamera");
                }
            } else {
                // the camera is not active at this point
                this.cameraActive = false;
                // update again the button content
                this.btnStartCamera.setText("Start Camera");
                // stop the timer
                this.stopAcquisition();
            }
        } else {
            alert.show();
        }
    }

    /**
     * ######################################################################
     * OnClick Action Untuk mencari Folder untuk menyimpan gambar
     * var:
     *
     */
    @FXML
    private void browsSaveFileOnClick(ActionEvent event
    ) {
        DirectoryChooser brows = new DirectoryChooser();
        brows.setInitialDirectory(imgDir);
        brows.setTitle("Buka Folder Data Training");
        imgDir = brows.showDialog(apGetImage.getScene().getWindow());
        if (imgDir != null) {
            txtSaveFileLocation.setText(imgDir.getAbsolutePath());
        }
    }

    /**
     * ######################################################################
     * OnCllick Action creating new folder
     * var:
     * File file : lable file location
     */
    @FXML
    private void createFolderOnClick(ActionEvent event
    ) {
        File file = new File(imgDir + "\\" + txtFolderName.getText());

        if (!file.exists() && !txtFolderName.getText().equals("")) {
            file.mkdir();
        } else {
            alert = new Alert(Alert.AlertType.ERROR);
            alert.setContentText("Directory Already Exist");
            alert.show();
        }
        imgDir = new File(imgDir.getAbsolutePath());
        txtIndexLabel.setText(String.valueOf(imgDir.list().length));
        imgLblDir = file;
    }

    @FXML
    private void getFramePointOnClick(MouseEvent event
    ) {
    }

    /**
     * ######################################################################
     *
     *
     */
    private void stopAcquisition() {
        if (this.timer != null && !this.timer.isShutdown()) {
            try {
                // stop the timer
                this.timer.shutdown();
                this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                // log any exception
                System.err.println(
                        "Exception in stopping the frame capture, trying to release the camera now... "
                        + e);
            }
        }
        if (this.capture.isOpened()) {
            // release the camera
            this.capture.release();
        }
    }

    /**
     * ######################################################################
     * update tampilan pada frame
     * var:
     *
     */
    private void updateImageView(ImageView view, Mat image) {
        Image imageToMat;
        imageToMat = Utils.mat2Image(image);
        Utils.onFXThread(view.imageProperty(), imageToMat);
    }

    /**
     * ######################################################################
     * On application close, stop the acquisition from the camera
     * var:
     *
     */
    protected void setClosed() {
        this.stopAcquisition();
    }

    /**
     * ######################################################################
     * Method untuk memperoleh Frame dari kamera
     *
     */
    private Mat grabFrame() {
        // init everything
        Mat frame = new Mat();
        // check if the capture is open
        if (this.capture.isOpened()) {
            try {
                // read the current frame
                this.capture.read(frame);

                // if the frame is not empty, process it
                if (!frame.empty()) {
                    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BayerBG2BGR);

                }

            } catch (Exception e) {
                // log the error
                System.err.println("Exception during the image elaboration: "
                        + e);
            }
        }
        return frame;
    }

    /**
     * ######################################################################
     * method untuk menyimpan gambar dalam kotak merah MainFrame
     *
     */
    private void start(Mat frame) {
        Core.flip(frame, frame, 1);
        Preprocessing.drawRect(frame);
        updateImageView(layarMain, frame);
        Mat hand = Preprocessing.getBox(frame.clone());
        //
        if (imgDir.exists() && imgLblDir.exists()) {
            if (i < 1800) {
                Imgcodecs.imwrite(imgLblDir.getAbsolutePath() + "\\" + txtFolderName.getText() + "_" + txtIndex.getText() + ".jpg", hand
                );
                i++;
                txtIndex.setText(String.valueOf(i));
            }
        } else {
            alert.setContentText("Directory Not Exist");
            alert.show();
        }

    }

    void setMainController(MainAppController aThis) {
        this.mainAppController = aThis;
    }
}
