package com.rc;

import static spark.Spark.before;
import static spark.Spark.exception;
import static spark.Spark.get;
import static spark.Spark.halt;
import static spark.Spark.post;
import static spark.Spark.staticFiles;
import static spark.Spark.webSocket;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.BlockingQueue;

import javax.servlet.MultipartConfigElement;

import org.json.JSONException;
import org.json.JSONObject;
import org.pac4j.core.client.Clients;
import org.pac4j.core.config.Config;
import org.pac4j.core.exception.CredentialsException;
import org.pac4j.core.profile.CommonProfile;
import org.pac4j.core.util.CommonHelper;
import org.pac4j.http.client.indirect.FormClient;
import org.pac4j.http.credentials.UsernamePasswordCredentials;
import org.pac4j.http.credentials.authenticator.UsernamePasswordAuthenticator;
import org.pac4j.http.credentials.authenticator.test.SimpleTestUsernamePasswordAuthenticator;
import org.pac4j.http.profile.HttpProfile;
import org.pac4j.sparkjava.CallbackRoute;
import org.pac4j.sparkjava.DefaultHttpActionAdapter;
import org.pac4j.sparkjava.RequiresAuthenticationFilter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.neuland.jade4j.JadeConfiguration;
import de.neuland.jade4j.template.ClasspathTemplateLoader;
import de.neuland.jade4j.template.TemplateLoader;
import spark.Filter ;
import spark.ModelAndView;
import spark.Request;
import spark.Response;
import spark.Route;
import spark.template.jade.JadeTemplateEngine;

public class WebServer {
	private static Logger log = LoggerFactory.getLogger(WebServer.class);

	private Model nn ;

	public WebServer() throws IOException {

		staticFiles.location(".");
		staticFiles.expireTime(600);

		webSocket("/messages", WebSocketServer.class);

		exception( Exception.class, (exception, request, response) -> {
			response.body( "<pre>" + exception.toString() + "</pre>" ) ; 
		});

		TemplateLoader templateLoader = new ClasspathTemplateLoader();
		JadeConfiguration jadeConfig = new JadeConfiguration() ;
		jadeConfig.setCaching( false ) ;
		jadeConfig.setTemplateLoader(templateLoader);

//		final IndirectBasicAuthClient basicAuthClient = new IndirectBasicAuthClient();
		final FormClient authClient = new FormClient();
		Path pwds = Paths.get("data/pwds") ;
		if( !Files.isReadable(pwds) ) pwds = Paths.get("pwds") ;
		
		authClient.setAuthenticator( new SimpleAuthenticator( pwds ) );
		authClient.setName( "Basic" );
		authClient.setLoginUrl( "/login-form" );
		authClient.setCallbackUrl( "/login" );
		authClient.setUsernameParameter( "uid" ); 
		authClient.setPasswordParameter( "pwd" );
		
		final Clients clients = new Clients( "/login", authClient ) ;
		final Config config = new Config(clients);
		config.setHttpActionAdapter(new DefaultHttpActionAdapter() );
		
		Filter filter = new RequiresAuthenticationFilter(config, "Basic") ;
		before("/", (re,rs) -> { 
			log.info( "Opening page {}", re.pathInfo() );
			if( !re.pathInfo().equals("/login") ) {
				filter.handle(re,rs) ; 
			}
		} );

		final Route callback = new CallbackRoute(config);		
		post( "/login", callback) ;

		get( "/", this::main, new JadeTemplateEngine( jadeConfig ) ) ;
		get( "/instructions", this::instructions, new JadeTemplateEngine( jadeConfig ) ) ;
		
		get( "/config.json", this::config ) ;
		
		get( "/login-form", this::index, new JadeTemplateEngine( jadeConfig ) ) ;
		
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
		if( nn == null ) {
			log.error( "Can't train a network that's not yet created." ); 
			halt( 200, "Error: Please create a network before training." ) ;
			return "Error: Please create a network before training." ;
		}
		BlockingQueue<String> op = nn.train( p ) ;
		String rc = "Nothing happened!" ;
		String s = op.take() ;
		while( s.length()>0 ) {
			rc = s ;
			WebSocketServer.send( s ) ; 
			s = op.take() ;
		}
		return rc ;
	}
	
	public Object uploadTest( Request request, Response response ) throws Exception {
		Path p = Files.createTempFile( "test", ".dat" ) ;
		log.info( "Saving to {}" , p ) ;
		Files.write( p, request.bodyAsBytes() ) ;
		String rc = nn.test( p ) ;
		return rc ;
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

	public Object create( Request request, Response response ) throws IOException {
		//		String config = request.body() ;
		MultipartConfigElement multipartConfigElement = new MultipartConfigElement("/create");
		request.raw().setAttribute("org.eclipse.jetty.multipartConfig", multipartConfigElement);

		log.info( "Received request to build {} network", request.queryParams( "layer-type" ) ) ;
		String layerType = request.queryParams( "layer-type" ) ;
		String tmp = request.queryParams( "num-inputs" ) ;
		if( tmp == null || tmp.length()==0 ) { WebSocketServer.send( "Mising number of inputs." ) ; halt( 400, "Missing num-inputs" ) ; }
		int numInputs = Integer.parseInt( tmp ) ;
		tmp = request.queryParams( "num-outputs" ) ;
		if( tmp == null || tmp.length()==0 ) { WebSocketServer.send( "Mising number of outputs." ) ; halt( 400, "Missing num-outputs" ) ; }
		int numOutputs = Integer.parseInt( tmp ) ;
		tmp = request.queryParams( "num-layers" ) ;
		if( tmp == null || tmp.length()==0 ) { WebSocketServer.send( "Mising number of layers." ) ; halt( 400, "Missing num-layers" ) ; }
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
		return "Created " + numLayers + " layer " + layerType + " network."  ;
	}

	public Object config( Request request, Response response ) {
		response.header( "Content-Type", "application/json" );
		return nn.getModel(0).getLayerWiseConfigurations().toJson() ;
	}

	public ModelAndView index( Request request, Response response ) {
		Map<String,Object> map = new HashMap<>() ;
		return new ModelAndView( map, "templates/index" )  ;
	}

	public ModelAndView login( Request request, Response response ) {
		return null ;
	}

	public ModelAndView main( Request request, Response response ) {
		Map<String,Object> map = new HashMap<>() ;
		return new ModelAndView( map, "templates/nnp" )  ;
	}


	public ModelAndView instructions( Request request, Response response ) {
		Map<String,Object> map = new HashMap<>() ;
		return new ModelAndView( map, "templates/instructions" )  ;
	}

	protected String getUsernameFromRequest( Request request ) {
		String h = request.headers( "Authorization" ) ;
		if( h==null ) return null ;
		String b64 = h.replace( "Basic ", "" ) ;
		String up = new String( Base64.getDecoder().decode( b64 ) ) ; 
		int ix = up.indexOf( ":" ) ;
		if( ix < 1 ) return null ;
		return up.substring(0,ix) ;
	}
}



class SimpleAuthenticator implements UsernamePasswordAuthenticator {

	protected static final Logger logger = LoggerFactory.getLogger(SimpleTestUsernamePasswordAuthenticator.class);

	private Map<String, String> pwds ;
	private Set<String> admins ;

	public SimpleAuthenticator( Path logins ) throws IOException {
		pwds = new HashMap<>() ;
		admins = new HashSet<>() ;

		Files.lines(logins)
		.map( s -> s.trim() )
		.filter( s -> s.length()>0 && s.charAt(0)!='#' )
		.map( s -> s.split( "," ) )
		.filter( s -> s.length>1 ) 
		.forEach( s -> { pwds.put( s[0],  s[1] ); if(s.length>2 && s[2].equalsIgnoreCase("admin") ) admins.add( s[0] ) ;}  )
		;
	}

	@Override
	public void validate(final UsernamePasswordCredentials credentials) {
		logger.info( "Validating" ) ; 
		if (credentials == null) {
			throwsException("No credential");
		}
		String username = credentials.getUsername();
		String password = credentials.getPassword();
		if (CommonHelper.isBlank(username)) {
			throwsException("Username cannot be blank");
		}
		if (CommonHelper.isBlank(password)) {
			throwsException("Password cannot be blank");
		}
		String storedpwd = pwds.get(username) ;
		if( storedpwd == null ) {
			throwsException("Unrecognized user");
		}
		if( !storedpwd.equals(password)) {
			throwsException("Username : '" + username + "' does not match password");
		}
		final HttpProfile profile = new HttpProfile();
		profile.setId(username);
		profile.addAttribute(CommonProfile.USERNAME, username);
		profile.addAttribute("ADMIN", admins.contains(username) );
		credentials.setUserProfile(profile);
	}

	protected void throwsException(final String message) {
		throw new CredentialsException(message);
	}
}

