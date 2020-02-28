package eu.h2020.helios_social.module.socialgraphmining;

import java.util.Map;

import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Edge;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;

public abstract class SocialGraphMiner {
	protected SocialGraphMiner(ContextualEgoNetwork contextualEgoNetwork) {
	}
	/** 
	 * Makes the graph miner aware that a user initiated an interaction to another user.
	 * @param interaction The new interaction the user initiates expressed in terms of the contextual ego network
	 */
    public abstract void newInteraction(Interaction interaction);
	/** 
	 * Makes the graph miner aware that a user received an interaction from another user with {@link #getModelParameters}.
	 * @param interaction A new interaction the user initiates expressed in terms of the contextual ego network
	 * @param neighborModelParameters The neighbor parameters to 
	 */
    public abstract void newInteraction(Interaction interaction, String neighborModelParameters, boolean isReply);
    /**
	 * Retrieves the parameters of the mining model that will be sent alongside the created interaction.
     * @param interaction The new interaction the user receives expressed in terms of the contextual ego network
     * @return A String serialization of model parameters.
     */
    public abstract String getModelParameters(Interaction interaction);
    /**
     * Predicts the weight of outgoing interactions
     * @return A map of edge weights.
     */
    public abstract Map<Edge, Double> predictOutgoingInteractions();
    /**
     * Provides various measures concerning the efficacy of predicting the given interaction using
     * the {@link #predictOutgoingInteractions()} method.
     * Needs be called before this interaction is registered (e.g. with {@link #newInteraction}.
     * Some of these measures assume binary (0/1) values and need be averaged across many interactions
     * to provide meaningful insights.
     * @param interaction The interaction to evaluate, expressed in terms of the contextual ego network
     * @return A hashmap between measures and values.
     */
    public abstract Map<String, Double> evaluate(Interaction interaction);
}