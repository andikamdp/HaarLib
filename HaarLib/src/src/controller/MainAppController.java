/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package src.controller;

import java.net.URL;

import java.util.ResourceBundle;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.scene.layout.AnchorPane;
import src.controller.HaarCobaController;
import java.io.IOException;
import java.util.logging.Logger;
import javafx.scene.Scene;

import java.util.logging.Level;
import src.MainApp;

/**
 * FXML Controller class
 *
 * @author Andika Mulyawan
 */
public class MainAppController implements Initializable {

    @FXML
    private Button btnHaar;
    @FXML
    private Button btnSvmTrain;
    @FXML
    private Button btnSvmPredict;
    @FXML
    private AnchorPane mainAppView;
    @FXML
    private Button btnSetImage;

    /**
     * Initializes the controller class.
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
    }
    private Stage haarWindow;

    /**
     * onClick action Membuka window baru Haar testing
     *
     */
    @FXML
    private void haarOpenWindowOnClick(ActionEvent event) {
        try {
            FXMLLoader loader = new FXMLLoader();
            loader.setLocation(MainApp.class.getResource("view/haarCoba.fxml"));
            AnchorPane root = loader.load();
            HaarCobaController controller = loader.getController();
            controller.setMainController(this);
            Scene scene = new Scene(root);
            haarWindow = new Stage();
            haarWindow.initOwner(mainAppView.getScene().getWindow());
            haarWindow.initModality(Modality.WINDOW_MODAL);
            haarWindow.setScene(scene);
            haarWindow.show();
        } catch (IOException iOException) {
            Logger.getLogger(MainAppController.class.getName()).
                    log(Level.SEVERE, null, iOException);
        }

    }

    /**
     * onClick action membuka window baru berisi gui training SVM
     *
     */
    @FXML
    private void svmTrainWindowOnClick(ActionEvent event) {
        try {
            FXMLLoader loader = new FXMLLoader();
            loader.setLocation(MainApp.class.getResource("view/SVMTrain.fxml"));
            AnchorPane root = loader.load();
            SVMTrainController controller = loader.getController();
            controller.setMainController(this);
            Scene scene = new Scene(root);
            haarWindow = new Stage();
            haarWindow.initOwner(mainAppView.getScene().getWindow());
            haarWindow.initModality(Modality.WINDOW_MODAL);
            haarWindow.setScene(scene);
            haarWindow.show();
        } catch (IOException iOException) {
            Logger.getLogger(MainAppController.class.getName()).
                    log(Level.SEVERE, null, iOException);
        }
    }

    /**
     * onClick action membuka window baru berisi live capture image
     *
     */
    @FXML
    private void svmPredictOnClick(ActionEvent event) {
        try {
            FXMLLoader loader = new FXMLLoader();
            loader.setLocation(MainApp.class.getResource("view/HandView.fxml"));
            AnchorPane root = loader.load();
            HandViewController controller = loader.getController();
            controller.setMainController(this);
            Scene scene = new Scene(root);
            haarWindow = new Stage();
            haarWindow.initOwner(mainAppView.getScene().getWindow());
            haarWindow.initModality(Modality.WINDOW_MODAL);
            haarWindow.setScene(scene);
            haarWindow.show();
        } catch (IOException iOException) {
            Logger.getLogger(MainAppController.class.getName()).
                    log(Level.SEVERE, null, iOException);
        }
    }

    /**
     * onClick action membuka window baru berisi predict pergambar
     */
    @FXML
    private void setImageViewOnClick(ActionEvent event) {
        try {
            FXMLLoader loader = new FXMLLoader();
            loader.setLocation(MainApp.class.getResource("view/SetImage.fxml"));
            AnchorPane root = loader.load();
            SetImageController controller = loader.getController();
            controller.setMainController(this);
            Scene scene = new Scene(root);
            haarWindow = new Stage();
            haarWindow.initOwner(mainAppView.getScene().getWindow());
            haarWindow.initModality(Modality.WINDOW_MODAL);
            haarWindow.setScene(scene);
            haarWindow.show();
        } catch (IOException iOException) {
            Logger.getLogger(MainAppController.class.getName()).
                    log(Level.SEVERE, null, iOException);
        }
    }

}
