package com.rc;

import static spark.Spark.exception;
import static spark.Spark.get;
import static spark.Spark.halt;
import static spark.Spark.post;
import static spark.Spark.staticFiles;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import javax.servlet.MultipartConfigElement;

import org.json.JSONException;
import org.json.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.neuland.jade4j.JadeConfiguration;
import de.neuland.jade4j.template.ClasspathTemplateLoader;
import de.neuland.jade4j.template.TemplateLoader;
import spark.ModelAndView;
import spark.Request;
import spark.Response;
import spark.template.jade.JadeTemplateEngine;

public class WebServer {
	private static Logger log = LoggerFactory.getLogger(WebServer.class);

	private Model nn ;
	
	public WebServer() {
		//this.nn = nn ;
		
		TemplateLoader templateLoader = new ClasspathTemplateLoader();
		JadeConfiguration jadeConfig = new JadeConfiguration() ;
		jadeConfig.setCaching( false ) ;
		jadeConfig.setTemplateLoader(templateLoader);

    	staticFiles.location(".");
    	staticFiles.expireTime(600);
    	
		exception( Exception.class, (exception, request, response) -> {
			response.body( "<pre>" + exception.toString() + "</pre>" ) ; 
		});
    	
		get( "/", this::home, new JadeTemplateEngine( jadeConfig ) ) ;
		get( "/config.json", this::config ) ;
		post( "/create", this::create ) ;
		post( "/upload-train", this::uploadTrain ) ;
		post( "/upload-test", this::uploadTest ) ;
		post( "/upload-config", this::uploadConfig ) ;
	}

	
	public Object uploadTrain( Request request, Response response ) throws Exception {
		log.info( "Train request" ); 
		Path p = Files.createTempFile( "train", ".dat" ) ;
		log.info( "Saving to {}" , p ) ;
		Files.write( p, request.bodyAsBytes() ) ;
		nn.setTrainDataFile( p ) ;
		nn.train() ;
		return "" ;
	}
	public Object uploadTest( Request request, Response response ) throws Exception {
		Path p = Files.createTempFile( "test", ".dat" ) ;
		log.info( "Saving to {}" , p ) ;
		Files.write( p, request.bodyAsBytes() ) ;
		nn.setTestDataFile( p ) ;
		return nn.test().stats() ;
	}
	
	public Object uploadConfig( Request request, Response response ) {
		String config = request.body() ;
		try { 
			new JSONObject( config );
			nn.addModel(config);
		} catch( JSONException fail ) {
			halt( 402, "Invalid JSON" ) ;
		}
		return "OK" ;
	}

	public Object create( Request request, Response response ) {
//		String config = request.body() ;
		MultipartConfigElement multipartConfigElement = new MultipartConfigElement("/create");
	    request.raw().setAttribute("org.eclipse.jetty.multipartConfig", multipartConfigElement);

		log.info( request.queryParams( "layer-type" ) ) ;
		String layerType = request.queryParams( "layer-type" ) ;
		String tmp = request.queryParams( "num-inputs" ) ;
		int numInputs = Integer.parseInt( tmp ) ;
		tmp = request.queryParams( "num-outputs" ) ;
		int numOutputs = Integer.parseInt( tmp ) ;
		tmp = request.queryParams( "num-layers" ) ;
		int numLayers = Integer.parseInt( tmp ) ;
		nn = null ;
		if( layerType.equals( "MLP") ) {
			nn = new MultiLayer() ;
		} else if( layerType.equals( "W2V") ) {
			nn = new TextAnalyzer() ;
		} else if( layerType.equals( "DBN") ) {
			nn = new DBN() ;
		} else if( layerType.equals( "DBNA") ) {
			nn = new DBNA() ;
		} else if( layerType.equals( "LSTM") ) {
			nn = new LSTM() ;
		}
		nn.createModelConfig(numLayers, numInputs, numOutputs);
		return "Created " + numLayers + " layer " + layerType + " network." ;
	}

	public Object config( Request request, Response response ) {
		response.header( "Content-Type", "application/json" );
		return nn.getModel(0).getLayerWiseConfigurations().toJson() ;
	}
	
	public ModelAndView home( Request request, Response response ) {
		Map<String,Object> map = new HashMap<>() ;
		return new ModelAndView( map, "templates/index" )  ;
	}
}
