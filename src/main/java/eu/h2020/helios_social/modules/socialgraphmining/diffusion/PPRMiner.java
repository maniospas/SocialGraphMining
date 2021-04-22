package eu.h2020.helios_social.modules.socialgraphmining.diffusion;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.core.contextualegonetwork.Utils;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.tensor.DenseTensor;

/**
 * This class implements a Personalized PageRank scheme, where each ego node's personalization
 * is a vector (modeled by a JGNN library Tensor) passed to the constructor and which is smoothed over
 * the decentralized social graph. Smoothing outcome is obtained through the method
 * {@link #getSmoothedPersonalization()}.
 * 
 * @author Emmanouil Krasanakis
 */
public class PPRMiner extends SocialGraphMiner {
	private String name;
	private double restartProbability = 0.1;
	private boolean personalizationAsGroundTruth = false;
	
	public PPRMiner(String name, ContextualEgoNetwork contextualEgoNetwork, Tensor personalization) {
		super(contextualEgoNetwork);
		if(name==null || name.isEmpty())
			Utils.error(new IllegalArgumentException());
		this.name = name;
		updatePersonalization(personalization);
	}
	
	/**
	 * Retrieves the restart probability of the equivalent random walk with restart scheme associated
	 * with personalized PageRank.
	 * @return The restart probability.
	 * @see #setRestartProbability(double)
	 */
	public synchronized double getRestartProbability() {
		return restartProbability;
	}
	
	/**
	 * Sets whether personalization should be considered as ground truth. If this is true and the personalization
	 * vector has a non-zero norm (i.e. has at least one non-zero element), then the outcome
	 * of {@link #getSmoothedPersonalization()} is forcefully snapped to the personalization vector. Making this
	 * depend on the norm helps deployment of models.
	 * @param personalizationAsGroundTruth A boolean value on whether personalization should be considered ground truth.
	 * @return <code>this</code> miner's instance.
	 */
	public synchronized PPRMiner setPersonalizationAsGroundTruth(boolean personalizationAsGroundTruth) {
		this.personalizationAsGroundTruth = personalizationAsGroundTruth;
		return this;
	}
	
	/**
	 * Sets the restart probability of the personalized PageRank scheme. Smaller values induce
	 * broader diffusion of predictions, i.e. many hops away in the social graph. The equivalent
	 * random walk with restart scheme has average random walk length equal to 1/(restart probability).
	 * <br>
	 * Suggested values to experiment with:<br>
	 * - 0.15 (used in older personalized PageRank papers)
	 * - 0.10 (default, used by graph neural networks to great success)
	 * - 0.01 (extremely long walks, suitable to detect communities of high radius from few examples)
	 * @param restartProbability The restart probability in the range (0,1) (default is 0.1).
	 * @return <code>this</code> miner's instance.
	 */
	public synchronized PPRMiner setRestartProbability(double restartProbability) {
		if(restartProbability<=0 || restartProbability>=1)
			Utils.error("Restart probabilty should be in the open range (0,1)");
		this.restartProbability = restartProbability;
		return this;
	}
	
	public synchronized PPRMiner updatePersonalization(Tensor personalization) {
		getContextualEgoNetwork()
			.getEgo()
			.getOrCreateInstance(getModuleName()+"personalization", ()->personalization.zeroCopy())
			.setToZero()
			.selfAdd(personalization);
		return this;
	}
	
	/**
	 * Retrieves the module's name used as prefix to identifiers for the {@link Node#getOrCreateInstance(String, Class)} 
	 * methods when retrieving data attached to nodes.
	 * @return The module name's prefix.
	 */
	protected String getModuleName() {
		return name;
	}
	
	/**
	 * Retrieves the personalization 
	 * @return The personalization vector.
	 * @see #updatePersonalization(Tensor)
	 */
	public Tensor getPersonalization() {
		return getContextualEgoNetwork()
				.getEgo()
				.getOrCreateInstance(getModuleName()+"personalization", DenseTensor.class);
	}
	
	/**
	 * 
	 * @return
	 */
	public synchronized Tensor getSmoothedPersonalization() {
		return getContextualEgoNetwork().getEgo().getOrCreateInstance(getModuleName()+"score", () -> getPersonalization().copy());
	}
	
	@Override
	public synchronized void newInteractionParameters(Interaction interaction, SocialGraphMinerParameters neighborModelParameters, InteractionType interactionType) {
		if(interaction.getEdge().getAlter()==null)
			return;
		Tensor neighborScore = (Tensor) neighborModelParameters.get("score");
		interaction
			.getEdge()
			.getAlter()
			.getOrCreateInstance(getModuleName()+"score", ()->neighborScore.zeroCopy())
			.setToZero()
			.selfAdd(neighborScore);
		
		//it's impossible for numNodes to be 0 at this point, since an interaction has occurred
		if(!personalizationAsGroundTruth || getPersonalization().norm()!=0) {
			int numNodes = interaction.getEdge().getContext().getNodes().size();
			Tensor score = getSmoothedPersonalization().setToZero().selfAdd(getPersonalization()).selfMultiply(numNodes*restartProbability/(1-restartProbability));
			for(Node node : interaction.getEdge().getContext().getNodes()) 
				score.selfAdd(node.getOrCreateInstance(getModuleName()+"score", ()->score.zeroCopy()));
			score.selfMultiply((1-restartProbability)/numNodes);
		}
		else
			getSmoothedPersonalization().setToZero().selfAdd(getPersonalization());
	}

	@Override
	public SocialGraphMinerParameters getModelParameterObject(Interaction interaction) {
		SocialGraphMinerParameters params = new SocialGraphMinerParameters();
		params.put("score", getSmoothedPersonalization());
		return params;
	}

	@Override
	public double predictNewInteraction(Context context, Node destinationNode) {
		throw new RuntimeException("PPRMiner is not meant to predict interactions");
	}

}
