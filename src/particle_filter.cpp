/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <assert.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).


	// set the number of particles
	num_particles = 100;
	particles.resize(num_particles);
	// initialize all weights to 1
	weights.resize(num_particles,1);
	// create three Gaussian noise generators
	random_device rd;
	default_random_engine gen(rd());

	normal_distribution<double> d_x(x,std[0]);
	normal_distribution<double> d_y(y,std[1]);
	normal_distribution<double> d_theta(theta,std[2]);

	// initialize the particles' values
	for( auto p : particles){
		p.x = d_x(gen);
		p.y = d_y(gen);
		p.theta = d_theta(gen);
		p.weight = 1;
	}

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// create random noise generators
	random_device rd;
	default_random_engine gen(rd());
	normal_distribution<double> d_x(0,std_pos[0]);
	normal_distribution<double> d_y(0,std_pos[1]);
	normal_distribution<double> d_theta(0,std_pos[2]);

	for(auto p:particles){
		if(yaw_rate < 0.0001){
			p.x += velocity * cos(p.theta) * delta_t;
			p.y += velocity * sin(p.theta) * delta_t;
		}else{
			p.x += velocity * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta))/yaw_rate;
			p.y += velocity * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t))/yaw_rate;
		}
		p.theta += yaw_rate*delta_t;
		//add noises
		p.x += d_x(gen);
		p.y += d_y(gen);
		p.theta += d_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double std_x = std_landmark[0];
	double std_y = std_landmark[1];
	double var_x = pow(std_x,2);
	double var_y = pow(std_y,2);

	for(uint i = 0; i<particles.size();i++){

		Particle & p=particles[i];

		vector<LandmarkObs> predicted;
		for(auto landmark:map_landmarks.landmark_list)
		{
			if(dist(p.x,p.y,landmark.x_f,landmark.y_f)<=sensor_range)
			{
				LandmarkObs l;
				l.x = landmark.x_f;
				l.y = landmark.y_f;
				predicted.push_back(l);
			}
		}
		//dataAssociation(predicted,observations);
		p.weight = 1;
		double cos_theta = cos(p.theta);
		double sin_theta = sin(p.theta);
		for(auto obs:observations){
			double obs_x_global = p.x + obs.x * cos_theta-obs.y * sin_theta;
			double obs_y_global = p.y + obs.x * sin_theta+obs.y * cos_theta;
			//nearst landmark
			LandmarkObs nlm;
			double nearst_distance = sensor_range;
			for(uint j=0; j< predicted.size(); j++){
				double distance =dist(obs_x_global,obs_y_global,predicted[j].x,predicted[j].y);
				if(distance < nearst_distance){
				   nlm = predicted[j];
				   nearst_distance = distance;
			    }
			}
			//get the bivariant PDF value, unnormalized weight
			p.weight *= 1/(std_x*std_y) * exp(-pow((obs_x_global - nlm.x),2)/var_x + pow((obs_y_global - nlm.y),2)/var_y);
		}

		weights[i] = p.weight;
	}
	double sum = accumulate(weights.begin(),weights.end(),0);
	for_each(weights.begin(),weights.end(),[sum](double x){return x/sum;});

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	random_device rd;
    default_random_engine gen(rd());
    discrete_distribution<int> sample(weights.begin(),weights.end());

    vector<Particle> newList(num_particles);
    for(uint i=0; i< num_particles;i++){
    	int rnd = sample(gen);
    	newList[i] = particles[rnd];
    }
    particles.swap(newList);
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
