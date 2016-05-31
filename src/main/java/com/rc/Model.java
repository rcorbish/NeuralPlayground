package com.rc;


import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.PathMatcher;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import org.canova.api.records.reader.RecordReader;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Updater;
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

	protected Path trainingData ;
	protected Path testData ;
	protected Path configDir ;

	int numInputs ;
	int labelIndices[] ;
	int numOutputs ;

	public Model() {
		this.trainingData = null ;
		this.testData = null ;
		this.configDir = null ;

		numInputs = 0 ;
		labelIndices = null ; 
		numOutputs = 0 ; 
	}
	
	public Model( Path trainingData, Path testData, Path configDir ) throws IOException {
		this.trainingData = trainingData ;
		this.testData = testData ;
		this.configDir = configDir ;

		numInputs = countInputsInDataFile(trainingData !=null ? trainingData : testData ) ;
		labelIndices = getLabelIndicesFromDataFile(trainingData !=null ? trainingData : testData ) ;
		numOutputs = countDistinctOutputsInDataFile(trainingData !=null ? trainingData : testData, labelIndices) ;

	}

	public void setTestDataFile( Path dataFile ) throws IOException {
		this.testData = dataFile ;

		numInputs = countInputsInDataFile(trainingData !=null ? trainingData : testData ) ;
		labelIndices = getLabelIndicesFromDataFile(trainingData !=null ? trainingData : testData ) ;
		numOutputs = countDistinctOutputsInDataFile(trainingData !=null ? trainingData : testData, labelIndices) ;
	}

	public void setTrainDataFile( Path dataFile ) throws IOException {
		this.trainingData = dataFile ;

		numInputs = countInputsInDataFile(trainingData !=null ? trainingData : testData ) ;
		labelIndices = getLabelIndicesFromDataFile(trainingData !=null ? trainingData : testData ) ;
		numOutputs = countDistinctOutputsInDataFile(trainingData !=null ? trainingData : testData, labelIndices) ;
	}

	protected List<MultiLayerNetwork> models = new ArrayList<>() ;

	public abstract void createModelConfig( int numLayers, int numInputs, int numOutputs ) ;
	
	protected void forEach( Consumer<MultiLayerNetwork> l ) { for( MultiLayerNetwork mln : models ) { l.accept(mln) ; } } ;
	protected MultiLayerNetwork getModel(int ix) { return models.get(ix); }
	protected int getNumModels() { return models.size() ; }
	protected void addModel( String jsonConf ) {
		MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson( jsonConf ) ;		
		addModel( conf ) ;
	}	
	protected void addModel( MultiLayerConfiguration conf ) {
		log.warn( "Clearing existing models - probably need to chnage this eventually" );
		models.clear(); 

		addModel(conf, null );
	}
	protected void addModel( MultiLayerConfiguration conf, INDArray params ) { 
		log.debug("Creating new model" ) ;
		MultiLayerNetwork mln = new MultiLayerNetwork( conf ) ;
		this.models.add( mln ) ; 
		mln.init();
		if( params != null ) {
			mln.setParameters(params);
		}
	}

	protected RecordReader getRecordReader( Path dataFile  ) {
		return null ;
	}

	abstract public void train() throws Exception ;
	abstract public Evaluation test() throws Exception ;

	public void saveModel( boolean saveUpdater ) throws IOException {

		if( configDir != null ) {
			if( !Files.isDirectory( configDir ) ) {
				Files.createDirectories( configDir) ;
			}
			
			for( int modelIndex = 0 ; modelIndex<getNumModels() ; modelIndex++ ) {
				//Write the network parameters:
				try(DataOutputStream dos = new DataOutputStream( Files.newOutputStream( getCoefficientFile(modelIndex,configDir)))){
					Nd4j.write(getModel(modelIndex).params(),dos);
				}

				//Write the network configuration:
				Files.write(getConfigFile(modelIndex,configDir), getModel(modelIndex).getLayerWiseConfigurations().toJson().getBytes());

				if( saveUpdater ) {
					try(ObjectOutputStream oos = new ObjectOutputStream(Files.newOutputStream(getUpdaterFile(modelIndex,configDir)))){
						oos.writeObject(getModel(modelIndex).getUpdater());
					}
				}
			}
		}
	}

	public void loadModel( boolean loadUpdater ) throws IOException, ClassNotFoundException {

		PathMatcher pm = FileSystems.getDefault().getPathMatcher( "glob:*." + CONFIG_FILE ) ;
		if( configDir != null ) {
			List<Path> configs = Files.list( configDir )
			.filter( p -> pm.matches( p.getFileName() ) )
			.collect( Collectors.toList() ) 
			;

			for( int modelIndex=0 ; modelIndex<configs.size() ; modelIndex++ ) {
				Path configFile = configs.get(modelIndex) ;
			
				if( !Files.isReadable( configFile ) || !Files.isRegularFile( configFile ) ) {
					throw new IOException( "Cannot read data from config file " + configFile + "." ) ;
				}
			//Load network configuration 
				MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson( new String( Files.readAllBytes(configFile) ) ) ;

				Path coefficientsFile = getCoefficientFile(modelIndex, configDir) ;
				if( Files.isReadable( coefficientsFile ) ) {
					try(DataInputStream dis = new DataInputStream( Files.newInputStream(coefficientsFile))){
						INDArray newParams = Nd4j.read(dis);
						addModel( conf, newParams );
					}
				} else {
					addModel( conf );
				}

				if( loadUpdater ) {
					Path updaterFile = getUpdaterFile(modelIndex,configDir) ;
					try(ObjectInputStream ois = new ObjectInputStream( Files.newInputStream(updaterFile))){
						getModel(modelIndex).setUpdater( (Updater) ois.readObject() ) ;
					}
				}
			}
		}
	}

	public int countDistinctOutputsInDataFile( Path dataFile, int []labelIndices ) throws IOException {
		try ( BufferedReader br = Files.newBufferedReader(dataFile ) ) {
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

	public int countInputsInDataFile( Path dataFile ) throws IOException {
		try ( BufferedReader br = Files.newBufferedReader(dataFile ) ) {
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

	public int[] getLabelIndicesFromDataFile( Path dataFile ) throws IOException {
		try ( BufferedReader br = Files.newBufferedReader(dataFile ) ) {
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

	protected Path getConfigFile( int modelIndex, Path saveDir ) {
		return saveDir.resolve( String.valueOf(modelIndex)+ "." + CONFIG_FILE ) ;
	}
	protected Path getCoefficientFile( int modelIndex, Path saveDir ) {
		return saveDir.resolve( String.valueOf(modelIndex)+ "." + COEFFICIENTS_FILE ) ;
	}
	protected Path getUpdaterFile( int modelIndex, Path saveDir ) {
		return saveDir.resolve( String.valueOf(modelIndex)+ "." + UPDATER_FILE ) ;
	}
}
