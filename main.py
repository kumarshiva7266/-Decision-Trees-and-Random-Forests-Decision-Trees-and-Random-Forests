import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QFileDialog, QSpinBox, QDoubleSpinBox, QTabWidget,
                            QMessageBox, QProgressBar, QGroupBox, QGridLayout,
                            QSplitter, QFrame, QScrollArea, QTextEdit, QLineEdit,
                            QCheckBox, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
import graphviz
from sklearn import tree
import seaborn as sns
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')

class StyledButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1976D2;
                transform: scale(1.05);
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)

class StyledComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QComboBox {
                border: 2px solid #BDBDBD;
                border-radius: 5px;
                padding: 8px;
                background: white;
                font-size: 12px;
            }
            QComboBox:hover {
                border-color: #2196F3;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                width: 12px;
                height: 12px;
                background: #2196F3;
                border-radius: 3px;
            }
        """)

class DecisionTreeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Decision Trees & Random Forests Analyzer")
        self.setGeometry(100, 100, 1600, 1000)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F5;
            }
            QLabel {
                font-size: 12px;
                color: #424242;
                font-weight: bold;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #BDBDBD;
                border-radius: 8px;
                margin-top: 15px;
                padding: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px;
                color: #2196F3;
            }
            QProgressBar {
                border: 2px solid #BDBDBD;
                border-radius: 5px;
                text-align: center;
                background-color: #F5F5F5;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 3px;
            }
        """)
        
        self.data = None
        self.X = None
        self.y = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create tabs
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #BDBDBD;
                border-radius: 8px;
                background: white;
            }
            QTabBar::tab {
                background: #E0E0E0;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #2196F3;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background: #BDBDBD;
            }
        """)
        layout.addWidget(tabs)
        
        # Data Loading Tab
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        
        # File loading section
        file_group = QGroupBox("Dataset Loading")
        file_layout = QVBoxLayout()
        
        # Sample datasets
        sample_group = QGroupBox("Sample Datasets")
        sample_layout = QGridLayout()
        
        self.sample_datasets = {
            "Iris": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
            "Wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
            "Breast Cancer": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
        }
        
        row = 0
        col = 0
        for name, url in self.sample_datasets.items():
            btn = StyledButton(f"Load {name}")
            btn.clicked.connect(lambda checked, u=url: self.load_sample_data(u))
            sample_layout.addWidget(btn, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1
        
        sample_group.setLayout(sample_layout)
        file_layout.addWidget(sample_group)
        
        # Custom dataset loading
        custom_group = QGroupBox("Custom Dataset")
        custom_layout = QVBoxLayout()
        
        self.load_btn = StyledButton("Load Custom Dataset")
        self.load_btn.clicked.connect(self.load_data)
        custom_layout.addWidget(self.load_btn)
        
        self.data_preview = QTextEdit()
        self.data_preview.setReadOnly(True)
        self.data_preview.setMaximumHeight(200)
        self.data_preview.setStyleSheet("""
            QTextEdit {
                border: 2px solid #BDBDBD;
                border-radius: 5px;
                padding: 5px;
                background: white;
            }
        """)
        custom_layout.addWidget(QLabel("Data Preview:"))
        custom_layout.addWidget(self.data_preview)
        
        custom_group.setLayout(custom_layout)
        file_layout.addWidget(custom_group)
        
        file_group.setLayout(file_layout)
        data_layout.addWidget(file_group)
        
        # Model Configuration Tab
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)
        
        # Model type selection
        model_group = QGroupBox("Model Configuration")
        model_config_layout = QGridLayout()
        
        model_config_layout.addWidget(QLabel("Model Type:"), 0, 0)
        self.model_type = StyledComboBox()
        self.model_type.addItems(["Decision Tree", "Random Forest"])
        self.model_type.currentIndexChanged.connect(self.update_model_params)
        model_config_layout.addWidget(self.model_type, 0, 1)
        
        # Parameters section
        model_config_layout.addWidget(QLabel("Max Depth:"), 1, 0)
        self.max_depth = QSpinBox()
        self.max_depth.setRange(1, 50)
        self.max_depth.setValue(5)
        model_config_layout.addWidget(self.max_depth, 1, 1)
        
        model_config_layout.addWidget(QLabel("Min Samples Split:"), 2, 0)
        self.min_samples_split = QSpinBox()
        self.min_samples_split.setRange(2, 50)
        self.min_samples_split.setValue(2)
        model_config_layout.addWidget(self.min_samples_split, 2, 1)
        
        # Random Forest specific parameters
        self.n_estimators_label = QLabel("Number of Trees:")
        self.n_estimators = QSpinBox()
        self.n_estimators.setRange(10, 1000)
        self.n_estimators.setValue(100)
        model_config_layout.addWidget(self.n_estimators_label, 3, 0)
        model_config_layout.addWidget(self.n_estimators, 3, 1)
        
        # Advanced options
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QGridLayout()
        
        self.cross_validation = QCheckBox("Use Cross-Validation")
        self.cross_validation.setChecked(True)
        advanced_layout.addWidget(self.cross_validation, 0, 0)
        
        self.grid_search = QCheckBox("Use Grid Search")
        advanced_layout.addWidget(self.grid_search, 0, 1)
        
        self.standardize = QCheckBox("Standardize Features")
        self.standardize.setChecked(True)
        advanced_layout.addWidget(self.standardize, 1, 0)
        
        advanced_group.setLayout(advanced_layout)
        model_config_layout.addWidget(advanced_group, 4, 0, 1, 2)
        
        model_group.setLayout(model_config_layout)
        model_layout.addWidget(model_group)
        
        # Training section
        train_group = QGroupBox("Training")
        train_layout = QVBoxLayout()
        
        self.train_btn = StyledButton("Train Model")
        self.train_btn.clicked.connect(self.train_model)
        train_layout.addWidget(self.train_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        train_layout.addWidget(self.progress_bar)
        
        train_group.setLayout(train_layout)
        model_layout.addWidget(train_group)
        
        # Results Tab
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        # Create splitter for results
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel for metrics
        metrics_group = QGroupBox("Model Metrics")
        metrics_layout = QVBoxLayout()
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setStyleSheet("""
            QTextEdit {
                border: 2px solid #BDBDBD;
                border-radius: 5px;
                padding: 5px;
                background: white;
                font-family: 'Consolas', monospace;
            }
        """)
        metrics_layout.addWidget(self.metrics_text)
        metrics_group.setLayout(metrics_layout)
        
        # Right panel for visualizations
        viz_group = QGroupBox("Visualizations")
        viz_layout = QVBoxLayout()
        
        # Create matplotlib figure for visualization
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        
        viz_group.setLayout(viz_layout)
        
        splitter.addWidget(metrics_group)
        splitter.addWidget(viz_group)
        results_layout.addWidget(splitter)
        
        # Add tabs to main widget
        tabs.addTab(data_tab, "Data Loading")
        tabs.addTab(model_tab, "Model Configuration")
        tabs.addTab(results_tab, "Results")
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Initialize model parameters
        self.update_model_params()
        
    def update_model_params(self):
        is_random_forest = self.model_type.currentText() == "Random Forest"
        self.n_estimators_label.setVisible(is_random_forest)
        self.n_estimators.setVisible(is_random_forest)
        
    def load_sample_data(self, url):
        try:
            self.statusBar().showMessage(f"Loading sample dataset...")
            self.data = pd.read_csv(url)
            
            # Update data preview
            preview_text = f"Dataset Shape: {self.data.shape}\n\n"
            preview_text += "First 5 rows:\n"
            preview_text += str(self.data.head())
            self.data_preview.setText(preview_text)
            
            QMessageBox.information(self, "Success", "Sample dataset loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading sample dataset: {str(e)}")
    
    def load_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Dataset", "", 
                                                 "CSV Files (*.csv);;All Files (*)")
        if file_name:
            try:
                self.data = pd.read_csv(file_name)
                self.statusBar().showMessage(f"Loaded dataset: {file_name}")
                
                # Update data preview
                preview_text = f"Dataset Shape: {self.data.shape}\n\n"
                preview_text += "First 5 rows:\n"
                preview_text += str(self.data.head())
                self.data_preview.setText(preview_text)
                
                QMessageBox.information(self, "Success", "Dataset loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading dataset: {str(e)}")
    
    def train_model(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return
            
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Prepare data (assuming last column is target)
            X = self.data.iloc[:, :-1]
            y = self.data.iloc[:, -1]
            
            # Standardize features if selected
            if self.standardize.isChecked():
                X = self.scaler.fit_transform(X)
            
            # Encode target variable if needed
            if not np.issubdtype(y.dtype, np.number):
                y = self.label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.progress_bar.setValue(20)
            
            # Train model
            if self.model_type.currentText() == "Decision Tree":
                self.model = DecisionTreeClassifier(
                    max_depth=self.max_depth.value(),
                    min_samples_split=self.min_samples_split.value(),
                    random_state=42
                )
            else:
                self.model = RandomForestClassifier(
                    n_estimators=self.n_estimators.value(),
                    max_depth=self.max_depth.value(),
                    min_samples_split=self.min_samples_split.value(),
                    random_state=42
                )
            
            # Grid search if selected
            if self.grid_search.isChecked():
                param_grid = {
                    'max_depth': [3, 5, 7, 10],
                    'min_samples_split': [2, 5, 10]
                }
                if self.model_type.currentText() == "Random Forest":
                    param_grid['n_estimators'] = [50, 100, 200]
                
                grid_search = GridSearchCV(self.model, param_grid, cv=5)
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
            else:
                self.model.fit(X_train, y_train)
            
            self.progress_bar.setValue(60)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation if selected
            if self.cross_validation.isChecked():
                cv_scores = cross_val_score(self.model, X, y, cv=5)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std() * 2
            else:
                cv_mean = accuracy
                cv_std = 0
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Update metrics text
            metrics_text = f"Model Performance Metrics:\n\n"
            metrics_text += f"Accuracy: {accuracy:.4f}\n"
            if self.cross_validation.isChecked():
                metrics_text += f"Cross-validation scores: {cv_mean:.4f} (+/- {cv_std:.4f})\n"
            if self.grid_search.isChecked():
                metrics_text += f"\nBest Parameters:\n{self.model.get_params()}\n"
            metrics_text += "\nClassification Report:\n"
            metrics_text += classification_report(y_test, y_pred)
            
            self.metrics_text.setText(metrics_text)
            self.progress_bar.setValue(80)
            
            # Visualize results
            self.figure.clear()
            
            if self.model_type.currentText() == "Decision Tree":
                # Tree visualization
                plt.subplot(121)
                plot_tree(self.model, feature_names=self.data.columns[:-1], 
                         class_names=str(self.model.classes_),
                         filled=True, rounded=True, fontsize=8)
                plt.title("Decision Tree Visualization")
                
                # Confusion matrix
                plt.subplot(122)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title("Confusion Matrix")
            else:
                # Feature importance plot
                plt.subplot(121)
                importances = self.model.feature_importances_
                indices = np.argsort(importances)
                plt.barh(range(len(indices)), importances[indices])
                plt.yticks(range(len(indices)), self.data.columns[indices])
                plt.title("Feature Importance")
                
                # Confusion matrix
                plt.subplot(122)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title("Confusion Matrix")
            
            self.canvas.draw()
            self.progress_bar.setValue(100)
            
            # Show results
            QMessageBox.information(self, "Training Complete", 
                                  f"Model trained successfully!\nAccuracy: {accuracy:.4f}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error training model: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show the main window
    window = DecisionTreeGUI()
    window.show()
    sys.exit(app.exec_()) 