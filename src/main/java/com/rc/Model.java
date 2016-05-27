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
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

abstract public class Model {
	private static Logger log = LoggerFactory.getLogger(Model.class);

	private static String COEFFICIENTS_FILE = "coefficients.bin" ;
	private static String CONFIG_FILE = "config.json" ;
	private static String UPDATER_FILE = "updater.bin" ;

	protected File trainingData ;
	protected File testData ;
	protected File configDir ;

	int numInputs ;
	int labelIndices[] ;
	int numOutputs ;

	public Model( File trainingData, File testData, File configDir ) throws IOException {
		this.trainingData = trainingData ;
		this.testData = testData ;
		this.configDir = configDir ;
		
		numInputs = countInputsInDataFile(trainingData !=null ? trainingData : testData ) ;
		labelIndices = getLabelIndicesFromDataFile(trainingData !=null ? trainingData : testData ) ;
		numOutputs = countDistinctOutputsInDataFile(trainingData !=null ? trainingData : testData, labelIndices) ;
		
	}
	
	public void setTestDataFile( File dataFile ) throws IOException {
		this.testData = dataFile ;
		
		numInputs = countInputsInDataFile(trainingData !=null ? trainingData : testData ) ;
		labelIndices = getLabelIndicesFromDataFile(trainingData !=null ? trainingData : testData ) ;
		numOutputs = countDistinctOutputsInDataFile(trainingData !=null ? trainingData : testData, labelIndices) ;
	}
	
	public void setTrainDataFile( File dataFile ) throws IOException {
		this.trainingData = dataFile ;
		
		numInputs = countInputsInDataFile(trainingData !=null ? trainingData : testData ) ;
		labelIndices = getLabelIndicesFromDataFile(trainingData !=null ? trainingData : testData ) ;
		numOutputs = countDistinctOutputsInDataFile(trainingData !=null ? trainingData : testData, labelIndices) ;
	}
	
	private MultiLayerNetwork model ;

	protected MultiLayerNetwork getModel() { return model; }
	protected void setModel( String jsonConf ) {
		setModel( MultiLayerConfiguration.fromJson( jsonConf ) ) ;
	}	
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

	abstract public void train() throws Exception ;
	abstract public Evaluation<String> test() throws Exception ;
	abstract public MultiLayerConfiguration createModelConfig() throws IOException ;
	
	public void saveModel( boolean saveUpdater ) throws IOException {

		if( configDir != null ) {
			if( !configDir.isDirectory() && configDir.exists() ) {
				throw new IOException( "Cannot save to a file. " + configDir.getAbsolutePath() + " must be a directory." ) ;
			}
			if( !configDir.isDirectory() && !configDir.mkdirs() ) {
				throw new IOException( "Unable to create save directory: " + configDir.getAbsolutePath() + "." ) ;
			}
	
			//Write the network parameters:
			try(DataOutputStream dos = new DataOutputStream(new FileOutputStream(getCoefficientFile(configDir)))){
				Nd4j.write(getModel().params(),dos);
			}
	
			//Write the network configuration:
			FileUtils.write(getConfigFile(configDir), getModel().getLayerWiseConfigurations().toJson());
			if( saveUpdater ) {
				try(ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(getUpdaterFile(configDir)))){
					oos.writeObject(getModel().getUpdater());
				}
			}
		}
	}

	public void loadModel( boolean loadUpdater ) throws IOException, ClassNotFoundException {

		if( configDir != null ) {
			File configFile = getConfigFile(configDir) ;
			if( !(configFile.canRead() && configFile.isFile()) ) {
				throw new IOException( "Cannot read data from config file " + configFile.getAbsolutePath() + "." ) ;
			}
			//Load network configuration from disk:
			MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(configFile));
	
	
			File coefficientsFile = getCoefficientFile(configDir) ;
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
				File updaterFile = getUpdaterFile(configDir) ;
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
