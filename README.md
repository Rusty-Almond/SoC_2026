# Project Overview
Most of the code is from the templates provided in the [project repository](https://github.com/ShubhSarin/FromMathToMagic) with some modifications as required.

## [Week 0](Week 0/)
Basic intro week where you got to learn the basics of python, numpy and pytorch, libraries used heavily later. To get a better understanding of making models via pytorch, a simple linear model was trained on synthetic data to fit a line from a set of points.

## Week 1
This week introduced the math behind the model we would implement in weeks 3,4,5 and 6. In particular, Gaussian distributions, Bayes' theorem and KL divergence were studied and the aim was to get an intuitive understanding of why do we need them and how is it useful. There was no submission for this week

## [Week 2](Week 2/)
This week focused heavily on what are latent variables, what is the reparametization trick and the derivation of the loss function used in our model(VAE) known as the ELBO(Evidence Lower Bound). Intuitive understanding is also important. The assignment was implementing the ELBO loss and the reparametrization trick and also the KL divergence. 

## [Week 3](Week 3/)
This week the VAE is fully trained on synthetic data points and the reconstructions, losses etc are seen. The effect of adding a $\beta$ to the KL divergence term can also be seen now. If $\beta$ is too high the KL term dominates and the VAE learns almost a perfect gaussian and thus outputs garbage but if it is too low, the KL term is very small and the reconstructions are very good but the generation is poor.

## [Week 4](Week 4/)
This week a VAE is trained but on the MNIST database and the regeneration and generation of digits is checked. The results can be seen in the jupyter  notebook

## [Week 5](Week 5/)
This week the VAE is trained on the harder CelebA dataset and the regeneration and generation of faces can be seen in the file.

## [Week 6](Week 6/)
This week we use the model from Week 5 and play around in its latent space, moving along a line and morphing a face, adding vectors like "smile" to a face, adding multiple attributes like "smile" and "young" to a face and also random walks.

## [Week 7](Week 7/)
From this week we start on another type of generative model called DDPM(Denoising Diffusion Probability Model). Here, you repeatedly add gaussian noise to an image and then train a neural network to generate an image from pure noise (the reverse process). The maths behind it is the focus of the week with topics like Markov chains, Gaussians and another reparametrization trick along with a new loss function and the manipulation to make it simpler to compute. Then there is code assignment to check the different schedulers, visualize destruction in forward process, seeing signal to noise ratio etc.

## [Week 8](Week 8/)
This week the Week 7 math is applied in code to create the forward process, schedulers and the loss function in reverse process.
