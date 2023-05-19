from cellpose import models
from cellpose.io import imread
from matplotlib import pyplot as plt
from matplotlib import patches
import os 
import sys
from PyQt5.QtWidgets import QFileDialog, QDialog,QPushButton,QVBoxLayout,QWidget,QApplication,QLabel,QProgressBar,QTextEdit
from PyQt5 import QtGui
import glob
import torch
import pims
import time
import numpy as np
import pandas as pd 
import trackpy as tp
from scipy.spatial import ConvexHull
from tqdm import tqdm
import cv2 as cv
dir = os.path.dirname(os.path.realpath(__file__))+'/'


def approxEllipseFromArea(img):
    ds={"y":0,"x":0,"theta":0,"width":0,"height":0}
    imgC=np.array(img,dtype=np.uint8)
    if np.max(imgC)!=0:
        cnt,h=cv.findContours(imgC, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        i=np.argmax([len(j) for j in cnt])
        el=cv.fitEllipse(cnt[i])
        ds['y']=el[0][0]
        ds['x']=el[0][1]
        ds['theta']=el[2]
        ds['width']=el[1][0]
        ds['height']=el[1][1]

    return ds



class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(800, 600)
        self.model=''
        self.folder=''
        self.mf=0
        self.ff=0
        self.sp=0
        self.consoleText=''
        #file=self.getFolder()
        self.folderButton = QPushButton('Open Project Folder')
        self.folderButton.clicked.connect(self.getFolder)
        self.folderLab=QLabel("No Folder Selected")
        self.modelButton=QPushButton('Select Model File')
        self.modelButton.clicked.connect(self.getModel)
        self.modelLab=QLabel("No Model Selected")
        self.startButton=QPushButton('Start')
        self.startButton.clicked.connect(self.start)
        self.progress=QProgressBar(self)
        self.console=QTextEdit(self.consoleText)
        self.showPlots=QPushButton("Show Plots")
        self.showPlots.clicked.connect(self.plot)
        layout = QVBoxLayout()
        layout.addWidget(self.folderButton)
        layout.addWidget(self.folderLab)
        layout.addWidget(self.modelButton)
        layout.addWidget(self.modelLab)
        layout.addWidget(self.startButton)
        layout.addWidget(self.progress)
        layout.addWidget(self.console)
        layout.addWidget(self.showPlots)

        self.setLayout(layout)

    def log(self,msg):
        self.console.append(msg)
        self.repaint()

    def getFolder(self):
        folder=QFileDialog.getExistingDirectory(self,'Image Directory')
        if folder:
            self.ff=1
            self.folderLab.setText(folder)
            self.folder=folder
        else:
            self.ff=0
            self.folderLab.setText("No Folder Selected")
    def getModel(self):
        [model,_]=QFileDialog.getOpenFileName(self,"Select Model",filter="*")
        if model:
            self.mf=1
            self.modelLab.setText(model)
            self.model=model
        else:
            self.mf=0
            self.modelLab.setText("No Model Selected")
    def start(self):
        if not self.mf or not self.ff:
            print(self.mf)
            print(self.ff)
            print( not self.mf or not self.ff)
            self.log("Model or Filder not selected")
            return
        self.log("Starting...")
        
        #time.sleep(1)
        self.md = models.CellposeModel(pretrained_model=self.model,gpu=True)
        self.log("Loaded Model")
        video=pims.open(self.folder+'/*.tif')
        self.frames=np.array(video)
        self.log("Loaded Frames")
        ch=[0,0]
        self.m=[]
        self.df=pd.DataFrame(columns=["y","x","theta","width","height","frame"])
        n=len(self.frames)
        self.progress.setMaximum(n)
        self.log("Running model, do not click anything")
        for i,f in enumerate(self.frames):
            masks, flows, styles = self.md.eval(f, diameter=None, channels=ch,progress=True)
            lab=np.unique(masks)
            for l in lab:
                if l!=0:
                    e=approxEllipseFromArea(masks==l)
                    e['frame']=i
                    self.df=pd.concat([self.df,pd.DataFrame(e,index=range(1))]).reset_index(drop=True)
            
            self.m.append(masks)
            self.progress.setValue(i)
            self.repaint()
            time.sleep(0.05)
        self.log("Finished")
        self.progress.setValue(n)
        self.sp=1

    def plot(self):
        if not self.sp:
            self.log("Analysis not finished")
            return
        
        fig,ax=plt.subplots(1,2,tight_layout=True,figsize=(10,5),subplot_kw={'projection':'polar'})
        fig2,ax2=plt.subplots(1,2,tight_layout=True,figsize=(10,5))
        theta=[]
        ratio=[]
        for i,r in self.df.query("frame=={}".format(0)).iterrows():
            theta.append(np.round(np.deg2rad(r["theta"]),2))
            ma=max((r["height"],r["width"]))
            mi=min((r["height"],r["width"]))
            ratio.append(ma/mi)
            e=patches.Ellipse((r['y'],r['x']), r['width'], r['height'],r['theta'])
            ax2[0].add_artist(e)
        ax[0].hist(theta,width=2*np.pi/20,bins=20)
        ax2[0].set_axis_off()
        ax2[0].imshow(self.frames[0])
        theta=[]
        ratio=[]
        for i,r in self.df.query("frame=={}".format(len(self.frames)-1)).iterrows():
            theta.append(np.round(np.deg2rad(r["theta"]),2))
            ma=max((r["height"],r["width"]))
            mi=min((r["height"],r["width"]))
            ratio.append(ma/mi)
            e=patches.Ellipse((r['y'],r['x']), r['width'], r['height'],r['theta'],alpha=0.5)
            ax2[1].add_artist(e)
        ax[1].hist(theta,width=2*np.pi/20,bins=20)
        ax2[1].set_axis_off()
        ax2[1].imshow(self.frames[-1])

        fig3,ax3=plt.subplots(1,1,tight_layout=True,figsize=(5,5))
        y=[]
        yerr=[]
        for i in range(len(self.frames)):
            y.append(self.df.query("frame=={}".format(i))['theta'].mean())
            yerr.append(self.df.query("frame=={}".format(i))['theta'].std())
        x=range(0,len(self.frames))
        ax3.errorbar(x,y,yerr=yerr,)
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Mean Angle (Deg)')
        ax3.set_ylim(0,360)
        #fig.savefig(dir+'radial.tif',dpi=800)
        #fig2.savefig(dir+'ellipse.tif',dpi=800)
        #fig3.savefig(dir+'time.tif',dpi=800)

        fig4,ax4=plt.subplots(1,1,tight_layout=True,figsize=(5,5))
        self.tracking=tp.link(self.df,15,memory=10)
        for p in self.tracking["particle"].unique():
            ps=self.tracking.query("particle=={}".format(p))
            ax4.plot(ps["frame"],ps["theta"])
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Theta')
        plt.show()
            


        



if __name__ == "__main__":
    app = QApplication(sys.argv)
    files=[]
    main = MainWindow()
    main.show()
    sys.exit(app.exec())
    '''
    md = models.CellposeModel(pretrained_model=dir+'models/CP_20220927_155811',gpu=True)

    #video=tifffile.imread('20221019_HUVECs_T3_P7_2_w3 Brightfield_s6_t35.TIF')
    #[nF,_,_]=video.shape
    #frames=pims.as_grey(pims.open(dir+'BF.avi'))
    video=pims.open(dir+'BF_1/*.tif')
    frames=np.array(video)
    ch=[0,0]
    m=[]
    df=pd.DataFrame(columns=["y","x","theta","width","height","frame"])
    for i,f in tqdm(enumerate(frames)):
        masks, flows, styles = md.eval(f, diameter=None, channels=ch,progress=True)
        lab=np.unique(masks)
        for l in lab:
            if l!=0:
                e=approxEllipseFromArea(masks==l)
                e['frame']=i
                df=pd.concat([df,pd.DataFrame(e,index=range(1))]).reset_index(drop=True)
        
        m.append(masks)


    fig,ax=plt.subplots(1,2,tight_layout=True,figsize=(10,5),subplot_kw={'projection':'polar'})
    fig2,ax2=plt.subplots(1,2,tight_layout=True,figsize=(10,5))
    theta=[]
    ratio=[]
    for i,r in df.query("frame=={}".format(0)).iterrows():
        theta.append(np.round(np.deg2rad(r["theta"]),2))
        ma=max((r["height"],r["width"]))
        mi=min((r["height"],r["width"]))
        ratio.append(ma/mi)
        e=patches.Ellipse((r['y'],r['x']), r['width'], r['height'],r['theta'])
        ax2[0].add_artist(e)
    ax[0].hist(theta,width=2*np.pi/20,bins=20)
    ax2[0].set_axis_off()
    ax2[0].imshow(frames[0])
    theta=[]
    ratio=[]
    for i,r in df.query("frame=={}".format(len(frames)-1)).iterrows():
        theta.append(np.round(np.deg2rad(r["theta"]),2))
        ma=max((r["height"],r["width"]))
        mi=min((r["height"],r["width"]))
        ratio.append(ma/mi)
        e=patches.Ellipse((r['y'],r['x']), r['width'], r['height'],r['theta'],alpha=0.5)
        ax2[1].add_artist(e)
    ax[1].hist(theta,width=2*np.pi/20,bins=20)
    ax2[1].set_axis_off()
    ax2[1].imshow(frames[-1])

    fig3,ax3=plt.subplots(1,1,tight_layout=True,figsize=(5,5))
    y=[]
    yerr=[]
    for i in range(len(frames)):
        y.append(df.query("frame=={}".format(i))['theta'].mean())
        yerr.append(df.query("frame=={}".format(i))['theta'].std())
    x=range(0,len(frames))
    ax3.errorbar(x,y,yerr=yerr,)
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Mean Angle (Deg)')
    ax3.set_ylim(0,360)
    fig.savefig(dir+'radial.tif',dpi=800)
    fig2.savefig(dir+'ellipse.tif',dpi=800)
    fig3.savefig(dir+'time.tif',dpi=800)

    fig4,ax4=plt.subplots(1,1,tight_layout=True,figsize=(5,5))
    tracking=tp.link(df,15,memory=10)
    for p in tracking["particle"].unique():
        ps=tracking.query("particle=={}".format(p))
        ax4.plot(ps["frame"],ps["theta"])
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Theta')

    plt.show()
    print(tracking.head())
    sys.exit(app.exec_())

'''