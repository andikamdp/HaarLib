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
import java.io.IOException;
import java.util.logging.Logger;
import javafx.scene.Scene;

import java.util.logging.Level;
import javafx.scene.layout.BorderPane;
import src.MainApp;

/**
 * FXML Controller class
 *
 * @author Andika Mulyawan
 */
public class MainAppController implements Initializable {

    @FXML
    private Button btnTrainingClassifierView;
    @FXML
    private Button btnLiveTranslationView;
    @FXML
    private AnchorPane mainAppView;
    @FXML
    private Button btnBuildImageView;

    /**
     * Initializes the controller class.
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TOD
    }
    private Stage stage;

    /**
     * onClick action membuka window baru berisi gui training SVM
     *
     */
    @FXML
    private void trainingClassifierViewOnClick(ActionEvent event) {
        try {
            FXMLLoader loader = new FXMLLoader();
            loader.setLocation(MainApp.class.getResource("view/TrainingClassifier.fxml"));
            BorderPane root = loader.load();
            TrainingClassifierController controller = loader.getController();
            controller.setMainController(this);
            Scene scene = new Scene(root);
            stage = new Stage();
            stage.initOwner(mainAppView.getScene().getWindow());
            stage.initModality(Modality.WINDOW_MODAL);
            stage.setScene(scene);
            stage.setTitle("Training Classifier");
            stage.show();
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
    private void liveTranslationViewOnClick(ActionEvent event) {
        try {
            FXMLLoader loader = new FXMLLoader();
            loader.setLocation(MainApp.class.getResource("view/LiveTranslation.fxml"));
            BorderPane root = loader.load();
            LiveTranslationController controller = loader.getController();
            controller.setMainController(this);
            Scene scene = new Scene(root);
            stage = new Stage();
            stage.initOwner(mainAppView.getScene().getWindow());
            stage.initModality(Modality.WINDOW_MODAL);
            stage.setScene(scene);
            stage.setTitle("Live Translation");
            stage.show();
        } catch (IOException iOException) {
            Logger.getLogger(MainAppController.class.getName()).
                    log(Level.SEVERE, null, iOException);
        }
    }

    /**
     * onClick action membuka window baru berisi predict pergambar
     */
    @FXML
    private void buildImageDatasetViewOnClick(ActionEvent event) {
        try {
            FXMLLoader loader = new FXMLLoader();
            loader.setLocation(MainApp.class.getResource("view/BuildImageDataset.fxml"));
            BorderPane root = loader.load();
            BuildImageDatasetController controller = loader.getController();
            controller.setMainController(this);
            Scene scene = new Scene(root);
            stage = new Stage();
            stage.initOwner(mainAppView.getScene().getWindow());
            stage.initModality(Modality.WINDOW_MODAL);
            stage.setScene(scene);
            stage.setTitle("Build Image Dataset");
            stage.show();
        } catch (IOException iOException) {
            Logger.getLogger(MainAppController.class.getName()).
                    log(Level.SEVERE, null, iOException);
        }
    }

}
