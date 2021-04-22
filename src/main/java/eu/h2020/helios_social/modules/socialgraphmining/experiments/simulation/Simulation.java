package eu.h2020.helios_social.modules.socialgraphmining.experiments.simulation;

import java.util.HashMap;
import java.util.Map.Entry;

import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.storage.NoStorage;
import eu.h2020.helios_social.modules.socialgraphmining.Measure;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;
import eu.h2020.helios_social.modules.socialgraphmining.heuristics.RepeatAndReplyMiner;
import mklab.JGNN.datasets.Dataset;
import eu.h2020.helios_social.modules.socialgraphmining.heuristics.DifferenceMiner;

/**
 * Performs a simulation that traverses over the interactions of a given {@link Dataset}
 * and assesses the outcome of mining them with an implemented type of miner passed to its
 * {@link #run(Dataset, Measure)} method.
 * 
 * Derived classes of this abstract one should implement its {@link #createMiner(ContextualEgoNetwork)}
 * method that creates the type of miner being tested.
 * 
 * @author Emmanouil Krasanakis
 */
public abstract class Simulation {
	private int maxInteractions = Integer.MAX_VALUE;
	private int avoidLastPredictions = 1;
	
	public Simulation() {}
	
	public abstract SocialGraphMiner createMiner(ContextualEgoNetwork cen);
	
	public Simulation setMaxInteractions(int maxInteractions) {
		this.maxInteractions = maxInteractions;
		return this;
	}
	
	public Simulation setLastPredictionToAvoid(int avoidLastPredictions) {
		this.avoidLastPredictions = avoidLastPredictions;
		return this;
	}
	
	protected SocialGraphMiner createDifferenceMiner(String deviceName) {
		ContextualEgoNetwork cen = ContextualEgoNetwork.createOrLoad(new NoStorage("NOFILESYSTEM\\"), deviceName, null);
		SocialGraphMiner miner = createMiner(cen);
		if(avoidLastPredictions==0)
			return miner;
		return new DifferenceMiner(miner, new RepeatAndReplyMiner(cen), avoidLastPredictions);
	}
	
	public Measure run(Dataset dataset, Measure measure) {
		HashMap<String, Device> devices = new HashMap<String, Device>();
		int currentInteraction = 0;
		for(Entry<String, String> interaction : dataset.getInteractions()) {
			String u = interaction.getKey();
			String v = interaction.getValue();
			if(u.equals(v))
				continue;
			if(!devices.containsKey(u))
				devices.put(u, new Device(createDifferenceMiner(u)));
			if(!devices.containsKey(v))
				devices.put(v, new Device(createDifferenceMiner(v)));
			measure.evaluateSend(devices.get(u).getMiner(),
					devices.get(u).getMiner().getContextualEgoNetwork().getCurrentContext(),
					devices.get(u).getMiner().getContextualEgoNetwork().getOrCreateNode(v));
			if(currentInteraction>=maxInteractions)
				break;
			devices.get(u).send(devices.get(v));
			currentInteraction++;
		}
		return measure;
	}
}
