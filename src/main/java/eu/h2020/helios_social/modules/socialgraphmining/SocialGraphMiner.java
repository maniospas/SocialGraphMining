package eu.h2020.helios_social.modules.socialgraphmining;

import java.util.HashMap;
import java.util.Set;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;

/**
 * Provides an abstraction of the basic capabilities and requirements of graph mining algorithms.
 * 
 * @author Emmanouil Krasanakis
 */
public abstract class SocialGraphMiner {
	public enum InteractionType {SEND, RECEIVE, RECEIVE_REPLY};
	
	public static class SocialGraphMinerParameters {
		private HashMap<String, Object> params;
		public SocialGraphMinerParameters() {
			params = new HashMap<String, Object>();
		}
		public Object get(String key) {
			return params.get(key);
		}
		public SocialGraphMinerParameters getNested(String key) {
			return (SocialGraphMinerParameters)get(key);
		}
		public void put(String key, Object value) {
			params.put(key, value);
		}
		public Set<String> getKeys() {
			return params.keySet();
		}
	}
	
	private ContextualEgoNetwork contextualEgoNetwork;
	private boolean sendPermision = true;
	
	protected SocialGraphMiner(ContextualEgoNetwork contextualEgoNetwork) {
		if(contextualEgoNetwork==null)
			throw new IllegalArgumentException();
		this.contextualEgoNetwork = contextualEgoNetwork;
	}
	public ContextualEgoNetwork getContextualEgoNetwork() {
		return contextualEgoNetwork;
	}
	public abstract void newInteractionParameters(Interaction interaction, SocialGraphMinerParameters neighborModelParameters, InteractionType interactionType);
	/** 
	 * Makes the graph miner aware that a user received an interaction from another user with {@link #getModelParameters}.
	 * @param interaction A new interaction the user initiates expressed in terms of the contextual ego network
	 * @param neighborModelParameters The neighbor parameters. May be null for when interactionType==SEND.
	 * @param interactionType The type of the interaction (SEND, RECEIVE or RECEIVE_REPLY corresponds to acknowledging the receive).
	 */
	public final void newInteraction(Interaction interaction, String neighborModelParameters, InteractionType interactionType) {
		Object parameters = neighborModelParameters==null?null:getContextualEgoNetwork().getSerializer().deserializeFromString(neighborModelParameters);
		newInteractionParameters(interaction, (SocialGraphMinerParameters) parameters, interactionType);
	}

	/**
	 * Sets whether the miner is permitted to send parameters when asked to, thus helping write seamless code, especially
	 * when multiple miner parameters are simultaneously send by {@link SwitchableMiner}. Setting send permisions to <code>true></code>
	 * makes miners work as intended, but setting them to <code>false</code> would force them to not send parameters
	 * that would improve recommendations for others.
	 * @param sendPermision Whether to allow the miner to send parameters.
	 * @return <code>this</code> miner
	 */
	public SocialGraphMiner setSendPermision(boolean sendPermision) {
		this.sendPermision = sendPermision;
		return this;
	}

    /**
	 * Retrieves the parameters of the mining model that will be sent alongside the created interaction.
	 * This method is wrapped by {@link #getModelParameters(Interaction)} to potentially not construct
	 * parameters at all, if {@link #setSendPermision(boolean)}.
     * @param interaction The new interaction the user receives expressed in terms of the contextual ego network
     * @return A {@link SocialGraphMinerParameters} object of miner parameters.
     */
	protected abstract SocialGraphMinerParameters constructModelParameterObject(Interaction interaction);
	
    /**
	 * Retrieves the parameters of the mining model that will be sent alongside the created interaction.
	 * Prefer use of {@link #getModelParameters(Interaction)} to guarantee correct serialization, though
	 * the two methods can used interchangeably.
	 * <b>This functionality can be reduced to sending a <code>null</code> by {@link #setSendPermision(boolean)}
	 * </b>
     * @param interaction The new interaction the user receives expressed in terms of the contextual ego network
     * @return A {@link SocialGraphMinerParameters} object of miner parameters.
     */
	public final SocialGraphMinerParameters getModelParameterObject(Interaction interaction) {
		if(!sendPermision)
			return null;
		return constructModelParameterObject(interaction);
	}
    /**
	 * Retrieves the parameters of the mining model that will be sent alongside the created interaction.
	 * This uses the contextual ego network's serializer to convert the outcome of {@link #getModelParameterObject(Interaction)} 
	 * to a string representation. The two methods can used interchangeably.
	 * <b>This functionality can be reduced to sending a serialized <code>null</code> by {@link #setSendPermision(boolean)}
	 * </b>
     * @param interaction The new interaction the user receives expressed in terms of the contextual ego network
     * @return A String serialization of model parameters.
     */
    public final String getModelParameters(Interaction interaction) {
    	return getContextualEgoNetwork().getSerializer().serializeToString(getModelParameterObject(interaction));
    }
    /**
     * Predicts the weight of performing a SEND interaction between the given context's ego and a destination node
     * within a given context. This method should typically return values in the [0,1] range, where higher values 
     * correspond to more likely interactions.
     * @param context The context in which to perform the prediction.
     * @param destinationNode The destination node of the interaction.
     * @return The weight of performing the interaction (higher values are more likely).
     */
    public abstract double predictNewInteraction(Context context, Node destinationNode);
    /**
     * Calls {@link #predictNewInteraction(Context, Node)} to score the likelihood of interacting with all nodes
     * of the given context.
     * @param context The context for which to recommend interactions.
     * @return A hash map of node scores (larger is more likely to occur.
     */
    public HashMap<Node, Double> recommendInteractions(Context context) {
    	HashMap<Node, Double> scores = new HashMap<Node, Double>();
    	Node ego = context.getContextualEgoNetwork().getEgo();
    	for(Node node : context.getNodes())
    		if(node!=ego) {
    			double value = predictNewInteraction(context, node);
    			if(Double.isFinite(value) && value!=0)
    				scores.put(node, value);
    		}
    	return scores;
    }
}