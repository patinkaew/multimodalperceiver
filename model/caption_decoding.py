import torch
import torch.nn as nn
import math

class VisionPerceiver(nn.Module):
    def __init__(self, input_dim, input_channels=3,
                 max_freq = 8, num_freq_bands = 4,
                 num_iterations = 1, num_transformer_blocks = 4,
                 num_latents = 32, latent_dim = 128,
                 cross_heads = 1, cross_dim_head = 8,
                 latent_heads = 2, latent_dim_head = 8,
                 num_classes = 10,
                 attn_dropout = 0., ff_dropout = 0.):

        super().__init__()

        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        input_channels *= 9
        # perceiver stacks
        self.layers = nn.ModuleList([])
        for i in range(num_iterations): # build each perceiver cell
            cell = nn.ModuleList([])
            # cross attention module
            cell.append(nn.LayerNorm(latent_dim))
            cell.append(nn.LayerNorm(input_dim * input_channels))
            cell.append(MultiHeadAttention(latent_dim, dim_head = cross_dim_head,
                                           num_heads = cross_heads, context_dim = input_channels,
                                           dropout = attn_dropout))
            # feed forward
            cell.append(nn.LayerNorm(latent_dim))
            cell.append(FeedForward(latent_dim, dropout = ff_dropout))

            # latent transformer
            latent_transformer = nn.ModuleList([])
            for j in range(num_transformer_blocks):
                latent_transformer_block = nn.ModuleList([])
                # self attention
                latent_transformer_block.append(nn.LayerNorm(latent_dim))
                latent_transformer_block.append(MultiHeadAttention(latent_dim, dim_head = latent_dim_head,
                                           num_heads = latent_heads, dropout = attn_dropout))
                # feed forward
                latent_transformer_block.append(nn.LayerNorm(latent_dim))
                latent_transformer_block.append(FeedForward(latent_dim, dropout = ff_dropout))
                latent_transformer.append(latent_transformer_block)
            cell.append(latent_transformer)

            self.layers.append(cell)

        self.to_logits = nn.Sequential(
                            nn.LayerNorm(latent_dim),
                            nn.Linear(latent_dim, num_classes))

    def forward(self, data, attn_mask = None, latent_init = None, seed = None):
        # flatten
        N, C, H, W = data.shape
        data = data.view(N, C, -1).transpose(1, 2)

        # encoding
        data = fourier_encode(data, self.max_freq, self.num_freq_bands)

        # determine the initial latent vector
        if latent_init is None:
            if seed is not None:
                torch.manual_seed(seed)
            latent_init = torch.randn(self.num_latents, self.latent_dim).unsqueeze(0).repeat([N, 1, 1])
            latent_init.requires_grad = False
        else:
            assert latent_init.shape == (num_latents, latent_dim)
            latent_init.unsqueeze(0).repeat([N, 1, 1])

        self.latent_init = nn.Parameter(latent_init)

        x = latent_init
        for cell in self.layers:
            # cross attention
            y = cell[1](data.reshape(N, -1))
            x = cell[2](cell[0](x), y.reshape(N, H*W, -1), attn_mask=attn_mask) + x
            # feed forward
            x = cell[4](cell[3](x)) + x
            # latent transformer
            for la2tent_transformer in cell[5]:
                # self attention
                x = latent_transformer[1](latent_transformer[0](x), attn_mask=attn_mask) + x
                # feed forward
                x = latent_transformer[3](latent_transformer[2](x)) + x

        x = x.mean(dim = -2)
        return x

def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != "<NULL>":
                words.append(word)
            if word == "<END>":
                break
        decoded.append(" ".join(words))
    if singleton:
        decoded = decoded[0]
    return decoded

def create_minibatch(data, batch_size=100, split="train"):
    split_size = data["%s_captions" % split].shape[0]
    mask = np.random.choice(split_size, batch_size)
    captions = data["%s_captions" % split][mask]
    image_idxs = data["%s_image_idxs" % split][mask]
    image_features = data["%s_features" % split][image_idxs]
    urls = data["%s_urls" % split][image_idxs]
    return captions, image_features, urls

class CaptioningSolverTransformer(object):
    """
    A CaptioningSolverTransformer encapsulates all the logic necessary for
    training Transformer based image captioning models.

    To train a model, you will first construct a CaptioningSolver instance,
    passing the model, dataset, and various options (learning rate, batch size,
    etc) to the constructor. You will then call the train() method to run the
    optimization procedure and train the model.

    After the train() method returns, the instance variable solver.loss_history
    will contain a list of all losses encountered during training.

    Example usage might look something like this:

    data = load_coco_data()
    model = MyAwesomeTransformerModel(hidden_dim=100)
    solver = CaptioningSolver(model, data,
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()


    A CaptioningSolverTransformer works on a model object that must conform to the following
    API:

      Inputs:
      - features: Array giving a minibatch of features for images, of shape (N, D
      - captions: Array of captions for those images, of shape (N, T) where
        each element is in the range (0, V].

      Returns:
      - loss: Scalar giving the loss
      - grads: Dictionary with the same keys as self.params mapping parameter
        names to gradients of the loss with respect to those parameters.
    """

    def __init__(self, model, data, idx_to_word, **kwargs):
        """
        Construct a new CaptioningSolver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data from load_coco_data

        Optional arguments:

        - learning_rate: Learning rate of optimizer.
        - batch_size: Size of minibatches used to compute loss and gradient during
          training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every print_every
          iterations.
        - verbose: Boolean; if set to false then no output will be printed during
          training.
        """
        self.model = model
        self.data = data

        # Unpack keyword arguments
        self.learning_rate = kwargs.pop("learning_rate", 0.001)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)

        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)
        self.optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        self._reset()

        self.idx_to_word = idx_to_word

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.loss_history = []


    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data
        captions, features, urls = create_minibatch(self.data, batch_size=self.batch_size, split="train")

        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        mask = captions_out != self.model._null

        t_features = torch.Tensor(features)
        t_captions_in = torch.LongTensor(captions_in)
        t_captions_out = torch.LongTensor(captions_out)
        t_mask = torch.LongTensor(mask)
        logits = self.model(t_features, t_captions_in)

        loss = self.transformer_temporal_softmax_loss(logits, t_captions_out, t_mask)
        self.loss_history.append(loss.detach().numpy())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.data["train_captions"].shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print(
                    "(Iteration %d / %d) loss: %f"
                    % (t + 1, num_iterations, self.loss_history[-1])
                )

            # At the end of every epoch, increment the epoch counter.
            epoch_end = (t + 1) % iterations_per_epoch == 0

    def transformer_temporal_softmax_loss(self, x, y, mask):
        """
        A temporal version of softmax loss for use in RNNs. We assume that we are
        making predictions over a vocabulary of size V for each timestep of a
        timeseries of length T, over a minibatch of size N. The input x gives scores
        for all vocabulary elements at all timesteps, and y gives the indices of the
        ground-truth element at each timestep. We use a cross-entropy loss at each
        timestep, summing the loss over all timesteps and averaging across the
        minibatch.

        As an additional complication, we may want to ignore the model output at some
        timesteps, since sequences of different length may have been combined into a
        minibatch and padded with NULL tokens. The optional mask argument tells us
        which elements should contribute to the loss.

        Inputs:
        - x: Input scores, of shape (N, T, V)
        - y: Ground-truth indices, of shape (N, T) where each element is in the range
             0 <= y[i, t] < V
        - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
          the scores at x[i, t] should contribute to the loss.

        Returns a tuple of:
        - loss: Scalar giving loss
        """
 
        N, T, V = x.shape

        x_flat = x.reshape(N * T, V)
        y_flat = y.reshape(N * T)
        mask_flat = mask.reshape(N * T)

        loss = torch.nn.functional.cross_entropy(x_flat,  y_flat, reduction='none')
        loss = torch.mul(loss, mask_flat)
        loss = torch.mean(loss)

        return loss