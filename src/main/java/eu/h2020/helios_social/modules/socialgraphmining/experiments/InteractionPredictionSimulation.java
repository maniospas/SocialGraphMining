package eu.h2020.helios_social.modules.socialgraphmining.experiments;

import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Utils;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.GNNMiner;
import eu.h2020.helios_social.modules.socialgraphmining.TF.TFMiner;
import eu.h2020.helios_social.modules.socialgraphmining.experiments.simulation.PeriodicReport;
import eu.h2020.helios_social.modules.socialgraphmining.experiments.simulation.Simulation;
import eu.h2020.helios_social.modules.socialgraphmining.heuristics.RandomMiner;
import eu.h2020.helios_social.modules.socialgraphmining.heuristics.RepeatAndReplyMiner;
import eu.h2020.helios_social.modules.socialgraphmining.measures.Average;
import eu.h2020.helios_social.modules.socialgraphmining.measures.HitRate;
import mklab.JGNN.datasets.Dataset;
import mklab.JGNN.datasets.Datasets;

public class InteractionPredictionSimulation {
	
	public static class GNNSimulation extends Simulation {
		@Override
		public SocialGraphMiner createMiner(ContextualEgoNetwork cen) {
			return new GNNMiner(cen)
					//.setEmbeddingExchangeProtocol(new CentralizedFederatedAveragingSimulation(1))
					//.setEmbeddingExchangeProtocol(new UncertainAvailabilitySimulation(1))
					.setRegularizationAbsorbsion(1)
					.setMinTrainingRelativeLoss(0.0001)
				    .setTrainingExampleDegradation(0.5)
				    .setTrainingExampleRemovalThreshold(0.0001*0.001)
				    .setDeniability(0, 0);
		}
	}
	public static class RandomSimulation extends Simulation {
		@Override
		public SocialGraphMiner createMiner(ContextualEgoNetwork cen) {
			return new RandomMiner(cen);
		}
	}
	public static class RepeatSimulation extends Simulation {
		@Override
		public SocialGraphMiner createMiner(ContextualEgoNetwork cen) {
			return new RepeatAndReplyMiner(cen);
		}
	}
	public static class TFSimulation extends Simulation {
		@Override
		public SocialGraphMiner createMiner(ContextualEgoNetwork cen) {
			return new TFMiner(cen);
		}
	}
	
	public static void main(String[] args) throws Exception {
		Utils.development = false;
		Dataset dataset = new Datasets.FRIENDS();
		PeriodicReport measure = new PeriodicReport(new Average(new HitRate(1)), 1000);
		(new RepeatSimulation())
			.setLastPredictionToAvoid(1)
			//.setMaxInteractions(20000)
			.run(dataset, measure);
	}
}
