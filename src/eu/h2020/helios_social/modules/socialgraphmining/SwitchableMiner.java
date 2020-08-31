package eu.h2020.helios_social.modules.socialgraphmining;

import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Set;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.core.contextualegonetwork.Utils;


/**
 * This class enables switching between different {@link SocialGraphMiner} implementations
 * for performing predictions while training them at the same time. After initializing it,
 * add miners using its {@link #createMiner(String, Class)} method. The active miner can
 * be selected using the {@link #setActiveMiner(String)} method.
 * 
 * @author Emmanouil Krasanakis
 */
public class SwitchableMiner extends SocialGraphMiner {
	private boolean locked = false;
	private SocialGraphMiner activeMiner;
	private HashMap<String, SocialGraphMiner> miners = new HashMap<String, SocialGraphMiner>();

	public SwitchableMiner(ContextualEgoNetwork contextualEgoNetwork) {
		super(contextualEgoNetwork);
	}
	
	/**
	 * Creates a miner of the given name for the given {@link SocialGraphMiner} class by calling its constructor that takes
	 * a social ego network as parameter. The created miner is stored in this object and is notified for new interactions
	 * alongside all other {@link #newInteraction(Interaction, String, InteractionType)} miners, regardless of whether it
	 * is set as the currently active miner by this class's respective method or not. Parameters of created miners are
	 * aggregated in this class through its {@link #getModelParameters(Interaction)} implementation and are de-aggregated at
	 * the receiving end of the received interaction.
	 * 
	 * @param minerName The name to be assigned at the miner
	 * @param minerClass The class of the created miner (should have a constructor with exactly one SocialEgoNetwork argument).
	 * @param SocialGraphMinerClass The implicitly understood class of the created miner.
	 * @return The created miner.
	 * @see #getMiner(String)
	 * @see #setActiveMiner(String)
	 */
	public <SocialGraphMinerClass extends SocialGraphMiner> SocialGraphMinerClass createMiner(String minerName, Class<SocialGraphMinerClass> minerClass) {
		if(locked)
			Utils.error("All createMiner() calls should have been performed immediately after initialization");
		try {
			SocialGraphMinerClass miner = minerClass.getConstructor(ContextualEgoNetwork.class).newInstance(getContextualEgoNetwork());
			if(miners.containsKey(minerName))
				Utils.error("Miner name already exists: "+minerName);
			miners.put(minerName, miner);
			return miner;
		}
		catch(Exception exception) {
			return Utils.error(exception, null);
		}
	}
	
	/**
	 * Retrieves the active miner.
	 * @return The active miner.
	 * @see #setActiveMiner(String)
	 */
	public SocialGraphMiner getActiveMiner() {
		return activeMiner;
	}
	
	/**
	 * Retrieves the names of all miners created through {@link #createMiner(String, Class)} calls.
	 * @return A set of created miner names
	 */
	public Set<String> getCreatedMinerNames() {
		return miners.keySet();
	}
	
	/**
	 * Retrieves a miner previously created with {@link #createMiner(String, Class)} by its given name.
	 * @param minerName The name of the created miner.
	 * @return The created miner.
	 * @see #getCreatedMinerNames()
	 */
	public SocialGraphMiner getMiner(String minerName) {
		SocialGraphMiner miner = miners.get(minerName); 
		if(miner==null) {
			Utils.error("No miner created with name: "+minerName);
			return null;
		}
		return miner;
	} 
	
	/**
	 * Gets a created miner with {@link #getMiner(String)} and, if such a miner is found,
	 * this is set as the active miner. The active miner is subsequently called to expose its outcome
	 * of {@link #predictNewInteraction(Context, Node)} and {@link #recommendInteractions(Context)}.
	 * Other functionalities are shared between all created miners.
	 * @param minerName The name of the miner to set as the 
	 * @return The new active miner
	 * @see #getActiveMiner()
	 * @see #createMiner(String, Class)
	 */
	public SocialGraphMiner setActiveMiner(String minerName) {
		activeMiner = getMiner(minerName);
		return activeMiner;
	} 

	@Override
	public void newInteractionFromMap(Interaction interaction, SocialGraphMinerParameters neighborModelParameters, InteractionType interactionType) {
		locked = true;
		for(String miner : neighborModelParameters.getKeys())
			if(miners.containsKey(miner))
				miners.get(miner).newInteractionFromMap(interaction, neighborModelParameters.getNested(miner), interactionType);
	}

	@Override
	public SocialGraphMinerParameters getModelParametersAsMap(Interaction interaction) {
		locked = true;
		SocialGraphMinerParameters ret = new SocialGraphMinerParameters();
		for(Entry<String, SocialGraphMiner> minerEntry : miners.entrySet())
			ret.put(minerEntry.getKey(), minerEntry.getValue().getModelParametersAsMap(interaction));
		return ret;
	}

	@Override
	public double predictNewInteraction(Context context, Node destinationNode) {
		locked = true;
		if(activeMiner==null)
			Utils.error("Must set an active miner before trying to predict interactions");
		return activeMiner.predictNewInteraction(context, destinationNode);
	}
	
	@Override
    public HashMap<Node, Double> recommendInteractions(Context context) {
		locked = true;
		if(activeMiner==null)
			Utils.error("Must set an active miner before trying to predict interactions");
		return activeMiner.recommendInteractions(context);
	}
	
}
