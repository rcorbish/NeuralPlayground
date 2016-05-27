package com.rc;

import static spark.Spark.exception;
import static spark.Spark.get;
import static spark.Spark.halt;
import static spark.Spark.post;
import static spark.Spark.staticFiles;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import org.json.JSONException;
import org.json.JSONObject;

import com.google.common.io.Files;

import de.neuland.jade4j.JadeConfiguration;
import de.neuland.jade4j.template.ClasspathTemplateLoader;
import de.neuland.jade4j.template.TemplateLoader;
import spark.ModelAndView;
import spark.Request;
import spark.Response;
import spark.template.jade.JadeTemplateEngine;

public class WebServer {

	private Model nn ;
	
	public WebServer( Model nn ) {
		this.nn = nn ;
		
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
		post( "/upload-train", this::uploadTrain ) ;
		post( "/upload-test", this::uploadTest ) ;
		post( "/upload-config", this::uploadConfig ) ;
	}

	
	public Object uploadTrain( Request request, Response response ) throws Exception {
		File f = File.createTempFile( "train", ".dat" ) ;
		Files.write( request.bodyAsBytes(), f ) ;
		nn.setTrainDataFile( f ) ;
		nn.train() ;
		return "" ;
	}
	public Object uploadTest( Request request, Response response ) throws Exception {
		File f = File.createTempFile( "test", ".dat" ) ;
		Files.write( request.bodyAsBytes(), f ) ;
		nn.setTestDataFile( f ) ;
		return nn.test().stats() ;
	}
	
	public Object uploadConfig( Request request, Response response ) {
		String config = request.body() ;
		try { 
			new JSONObject( config );
			nn.setModel(config);
		} catch( JSONException fail ) {
			halt( 402, "Invalid JSON" ) ;
		}
		return "OK" ;
	}

	public Object config( Request request, Response response ) {
		response.header( "Content-Type", "application/json" );
		return nn.getModel().getLayerWiseConfigurations().toJson() ;
	}
	
	public ModelAndView home( Request request, Response response ) {
		Map<String,Object> map = new HashMap<>() ;
		return new ModelAndView( map, "templates/index" )  ;
	}
}
