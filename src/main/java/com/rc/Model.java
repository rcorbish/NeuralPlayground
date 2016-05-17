package com.rc;


import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.io.FileUtils;
import org.canova.api.records.reader.RecordReader;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

abstract public class Model {
	private static Logger log = LoggerFactory.getLogger(Model.class);

	private static String COEFFICIENTS_FILE = "coefficients.bin" ;
	private static String CONFIG_FILE = "config.json" ;
	private static String UPDATER_FILE = "updater.bin" ;

	private MultiLayerNetwork model ;

	protected MultiLayerNetwork getModel() { return model; }
	protected void setModel( MultiLayerConfiguration conf ) {
		setModel(conf, null );
	}
	protected void setModel( MultiLayerConfiguration conf, INDArray params ) { 
		log.debug("Creating new model" ) ;
		this.model = new MultiLayerNetwork( conf ) ; 
		this.model.init();
		if( params != null ) {
			model.setParameters(params);
		}
	}
	
	protected RecordReader getRecordReader( File dataFile  ) {
		return null ;
	}

	abstract public DataSet train( File dataFile ) throws Exception ;
	abstract public DataSet test( File dataFile ) throws Exception ;
	abstract public MultiLayerConfiguration createModelConfig( File dataFile ) throws IOException ;
	
	public void plot( DataSet ds ) {
	       //Plot the data
        double xMin = -1.5;
        double xMax = 2.5;
        double yMin = -1;
        double yMax = 1.5;

        //Let's evaluate the predictions at every point in the x/y input space, and plot this in the background
        int nPointsPerAxis = 100;
        double[][] evalPoints = new double[nPointsPerAxis*nPointsPerAxis][2];
        int count = 0;
        for( int i=0; i<nPointsPerAxis; i++ ){
            for( int j=0; j<nPointsPerAxis; j++ ){
                double x = i * (xMax-xMin)/(nPointsPerAxis-1) + xMin;
                double y = j * (yMax-yMin)/(nPointsPerAxis-1) + yMin;

                evalPoints[count][0] = x;
                evalPoints[count][1] = y;

                count++;
            }
        }

        INDArray allXYPoints = Nd4j.create(evalPoints);
        INDArray predictionsAtXYPoints = model.output(allXYPoints);

        INDArray testPredicted = model.output(ds.getFeatures());
        PlotUtil.plotTestData(ds.getFeatures(), ds.getLabels(), testPredicted, allXYPoints, predictionsAtXYPoints, nPointsPerAxis);		
	}

	public void saveModel( File saveDir, boolean saveUpdater ) throws IOException {

		if( !saveDir.isDirectory() && saveDir.exists() ) {
			throw new IOException( "Cannot save to a file. " + saveDir.getAbsolutePath() + " must be a directory." ) ;
		}
		if( !saveDir.isDirectory() && !saveDir.mkdirs() ) {
			throw new IOException( "Unable to create save directory: " + saveDir.getAbsolutePath() + "." ) ;
		}

		//Write the network parameters:
		try(DataOutputStream dos = new DataOutputStream(new FileOutputStream(getCoefficientFile(saveDir)))){
			Nd4j.write(getModel().params(),dos);
		}

		//Write the network configuration:
		FileUtils.write(getConfigFile(saveDir), getModel().getLayerWiseConfigurations().toJson());
		if( saveUpdater ) {
			try(ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(getUpdaterFile(saveDir)))){
				oos.writeObject(getModel().getUpdater());
			}
		}
	}

	public void loadModel( File saveDir, boolean loadUpdater ) throws IOException, ClassNotFoundException {

		File configFile = getConfigFile(saveDir) ;
		if( !(configFile.canRead() && configFile.isFile()) ) {
			throw new IOException( "Cannot read data from config file " + configFile.getAbsolutePath() + "." ) ;
		}
		//Load network configuration from disk:
		MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(configFile));


		File coefficientsFile = getCoefficientFile(saveDir) ;
		if( !(configFile.canRead() && configFile.isFile()) ) {
			throw new IOException( "Cannot read data from coefficients file " + coefficientsFile.getAbsolutePath() + "." ) ;
		}

		//Load parameters from disk:
		INDArray newParams;
		try(DataInputStream dis = new DataInputStream(new FileInputStream(coefficientsFile))){
			newParams = Nd4j.read(dis);
		}


		//Create a MultiLayerNetwork from the saved configuration and parameters
		setModel( confFromJson, newParams );

		if( loadUpdater ) {
			File updaterFile = getUpdaterFile(saveDir) ;
			if( !(updaterFile.canRead() && configFile.isFile()) ) {
				throw new IOException( "Cannot read data from updater file " + updaterFile.getAbsolutePath() + "." ) ;
			}
			org.deeplearning4j.nn.api.Updater updater;
			try(ObjectInputStream ois = new ObjectInputStream(new FileInputStream(updaterFile))){
				updater = (org.deeplearning4j.nn.api.Updater) ois.readObject();
			}
			getModel().setUpdater(updater);
		}
	}

	public int countDistinctOutputsInDataFile( File dataFile, int []labelIndices ) throws IOException {
		try ( BufferedReader br = new BufferedReader( new FileReader( dataFile ) ) ) {
			String s = br.readLine() ;
			Set<String> outputs = new HashSet<>() ;
			
			StringBuilder key = new StringBuilder() ;
			while( (s=br.readLine()) != null ) {
				String cols[] = s.split(",") ;
				key.setLength(0);
				for( int i=0 ; i<labelIndices.length ; i++ ) {
					if( i>0 ) key.append( "|" ) ;
					key.append( cols[labelIndices[i]] ) ;
				}
				outputs.add( key.toString() ) ;
			}
			return outputs.size() ;
		}
	}

	public int countInputsInDataFile( File dataFile ) throws IOException {
		try ( BufferedReader br = new BufferedReader( new FileReader( dataFile ) ) ) {
			String hdrs = br.readLine() ;			
			String cols[] = hdrs.split( "," ) ;
			List<Integer> labels = new ArrayList<>() ;
			for( int i=0 ; i<cols.length ; i++ ) {
				if( cols[i].toLowerCase().contains("label") )  {
					labels.add( i ) ;
				}
			}
			if( labels.isEmpty() ) {
				labels.add( 0 ) ;
			}
			return cols.length - labels.size() ;
		}
	}

	public int[] getLabelIndicesFromDataFile( File dataFile ) throws IOException {
		try ( BufferedReader br = new BufferedReader( new FileReader( dataFile ) ) ) {
			String hdrs = br.readLine() ;			
			String cols[] = hdrs.split( "," ) ;
			List<Integer> labels = new ArrayList<>() ;
			for( int i=0 ; i<cols.length ; i++ ) {
				if( cols[i].toLowerCase().contains("label") )  {
					labels.add( i ) ;
				}
			}
			if( labels.isEmpty() ) labels.add( 0 ) ;
			int rc[] = new int[ labels.size() ] ;
			for( int i=0 ; i<labels.size() ; i++ ) {
				rc[i] = labels.get(i) ;
			}
			return rc ;
		} 
	}

	protected File getConfigFile( File saveDir ) {
		return new File( saveDir,CONFIG_FILE ) ;
	}
	protected File getCoefficientFile( File saveDir ) {
		return new File( saveDir,COEFFICIENTS_FILE ) ;
	}
	protected File getUpdaterFile( File saveDir ) {
		return new File( saveDir,UPDATER_FILE ) ;
	}
}
