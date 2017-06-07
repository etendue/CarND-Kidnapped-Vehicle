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
  num_particles = 200;
  particles.resize(num_particles);
  // initialize all weights to 1
  weights.resize(num_particles, 1);
  // create three Gaussian noise generators
  random_device rd;
  default_random_engine gen(rd());

  normal_distribution<double> d_x(x, std[0]);
  normal_distribution<double> d_y(y, std[1]);
  normal_distribution<double> d_theta(theta, std[2]);

  // initialize the particles' values
  for (auto &p : particles) {
    p.x = d_x(gen);
    p.y = d_y(gen);
    p.theta = d_theta(gen);
    p.weight = 1;
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  // create random noise generators
  random_device rd;
  default_random_engine gen(rd());
  normal_distribution<double> d_x(0, std_pos[0]);
  normal_distribution<double> d_y(0, std_pos[1]);
  normal_distribution<double> d_theta(0, std_pos[2]);

  for (auto &p : particles) {
    //motion transition
    if (yaw_rate < 0.0001) {  // Numeric stability check "division by Zero"
      p.x += velocity * cos(p.theta) * delta_t;
      p.y += velocity * sin(p.theta) * delta_t;
    } else {
      p.x += velocity * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta))
          / yaw_rate;
      p.y += velocity * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t))
          / yaw_rate;
    }
    p.theta += yaw_rate * delta_t;
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
  //process iterate each particles
  for (auto &p : particles) {
    // associate the observation with landmark on map.
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    double cos_theta = cos(p.theta);
    double sin_theta = sin(p.theta);
    for (auto obs : observations) {
      //transform observation landmark from vehicle to MAP coordinate
      double obs_x_global = p.x + obs.x * cos_theta - obs.y * sin_theta;
      double obs_y_global = p.y + obs.x * sin_theta + obs.y * cos_theta;
      //nearest landmark
      int land_mark_id = 0;
      double nearst_distance = numeric_limits<double>::max();
      for (auto &lm : predicted) {
        double distance = dist(obs_x_global, obs_y_global, lm.x, lm.y);
        if (distance < nearst_distance) {
          land_mark_id = lm.id;
          nearst_distance = distance;
        }
      }
      associations.push_back(land_mark_id);
      sense_x.push_back(obs_x_global);
      sense_y.push_back(obs_y_global);
    }
    p = SetAssociations(p, associations, sense_x, sense_y);
  }


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
  double var_x = pow(std_x, 2);
  double var_y = pow(std_y, 2);

  //get best particle
  Particle& best = *max_element(
      particles.begin(), particles.end(),
      [](const Particle &a,const Particle &b) {return a.weight <b.weight;});

  //convert map to landmark list
  vector<LandmarkObs> predicted;

  // confine the landmark inside the a range of 2 times sensor range + noise, i.e. diameter of sensor range

  double range = (sensor_range + std_x + std_y) * 2;

  for (uint i = 0; i < map_landmarks.landmark_list.size(); i++) {
    LandmarkObs l;
    l.x = map_landmarks.landmark_list[i].x_f;
    l.y = map_landmarks.landmark_list[i].y_f;
    l.id = map_landmarks.landmark_list[i].id_i;

    //filter out the landmarks outside the sensor range.
    if (dist(best.x, best.y, l.x, l.y) < range)
      predicted.push_back(l);
  }

  // associate the observation to landmark ids;
  dataAssociation(predicted, observations);



	//process iterate each particles
  double total_weight = 0;

  for (uint i = 0; i < particles.size(); i++) {

    Particle &p = particles[i];

    double log_weight = 0;
    for (uint j = 0; j < p.associations.size(); j++) {
      double sens_x = p.sense_x[j];
      double sens_y = p.sense_y[j];
      //associated landmark
      //the Land_mark id is from 1..N, for index minus 1.
      double x_mean = map_landmarks.landmark_list[p.associations[j] - 1].x_f;
      double y_mean = map_landmarks.landmark_list[p.associations[j] - 1].y_f;

      //get the bivariant PDF value, unnormalized log probability.
      //calculate log weight for numeric stability, as
      log_weight -= pow((sens_x - x_mean), 2) / var_x
          + pow((sens_y - y_mean), 2) / var_y;
    }
    //update the weight list
    p.weight = exp(log_weight);
    weights[i] = p.weight;
    total_weight += p.weight;
  }

  //if none of the particle is good enough, then the total_weight is zero. We have problem here.
  assert(total_weight != 0);
  //normalization of weights
  for_each(weights.begin(), weights.end(),
           [total_weight](double &w) {w/=total_weight;});
  for_each(particles.begin(), particles.end(),
           [total_weight](Particle &p) {p.weight/=total_weight;});
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  random_device rd;
  default_random_engine gen(rd());
  discrete_distribution<int> sample(weights.begin(), weights.end());

  vector<Particle> newList(num_particles);
  for (uint i = 0; i < num_particles; i++) {
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
