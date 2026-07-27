import torch
import torch.nn as nn
import torch.nn.functional as F



class A3CNetwork(nn.Module):
    def __init__(self, observation_size, num_actions):
        super(A3CNetwork, self).__init__()
        self.hidden_size=256
        self.num_layers = 1
        self.goal_Repr_Size= 12
        # CNN layers for processing the observation maps
        self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1),  
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=self.hidden_size - self.goal_Repr_Size, kernel_size=3, stride=1, padding=1)
                    )

        # Goal Position Processing
        self.goal_fc = nn.Linear(3, self.goal_Repr_Size)  # Match GOAL_REPR_SIZE

        # Fully Connected Layers 
        self.cnn_output_size = self._get_cnn_output_size(observation_size)
        self.fc1 = nn.Linear(self.cnn_output_size + self.goal_Repr_Size, self.hidden_size)
       
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)  # Matches `h2`
        self.dropout2 = nn.Dropout(p=0.3) 
        # Residual projection to match fc output
        self.residual_proj = nn.Linear(self.cnn_output_size + self.goal_Repr_Size, self.hidden_size)

        # LSTM layer for handling temporal dependencies
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)

        # Fully connected layers for policy, value, blocking, and on_goal outputs
        
        self.policy_layer = nn.Sequential(
                                        nn.Linear(self.hidden_size, self.hidden_size // 2),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_size // 2, num_actions)
                                        )

        self.value_layer = nn.Linear(self.hidden_size, 1)
        self.blocking_layer = nn.Linear(self.hidden_size, 1)
        self.on_goal_layer = nn.Linear(self.hidden_size, 1)
        
        self._initialize_weights()
    
        
    def _get_cnn_output_size(self, observation_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, observation_size, observation_size)
            dummy_output = self.cnn(dummy_input)
            return dummy_output.view(1, -1).size(1)

    def _initialize_weights(self):
        print("Initializing network weights...")

        # Initialize CNN layers
        for i, m in enumerate(self.cnn):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize Goal Processing Layer
        nn.init.xavier_uniform_(self.goal_fc.weight)
        nn.init.constant_(self.goal_fc.bias, 0)

        # Initialize Fully Connected Layers Before LSTM
        for name, layer in [("FC1", self.fc1), ("FC2", self.fc2)]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu', a=0.01)
            nn.init.constant_(layer.bias, 0)

        # Initialize LSTM layer
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                param.data.fill_(0)  # Default to zero
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1)  # Set forget gate bias to 1

        # Initialize Fully Connected Output Layers
        for name, layer in [("Value", self.value_layer), ("Blocking", self.blocking_layer), ("On Goal", self.on_goal_layer)]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='linear')
            nn.init.constant_(layer.bias, 0.0)

        # ✅ Initialize New Policy Layer (MLP)
        # Initialize final policy layer for logits
        for i, layer in enumerate(self.policy_layer):
            if isinstance(layer, nn.Linear):
                if i == 0:
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                else:
                    # Use Xavier initialization to ensure logits start balanced
                    nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)  # Ensure no initial bias

        # print("All network weights initialized successfully!")

    def forward(self, input_to_nn, goal_pos_to_nn,state_in,training=True):
        # Process the observation maps with CNN
        cnn_output = self.cnn(input_to_nn)
        cnn_output = cnn_output.view(cnn_output.size(0), -1)  # Flatten the CNN output
       
        cnn_output = F.leaky_relu(cnn_output, negative_slope=0.01)  #Apply Leaky ReLU AFTER flattening
       
        # Concatenate the CNN output with the goal position information
        # Goal Feature Extraction
        goal_layer = self.goal_fc(goal_pos_to_nn)
        
        # Concatenate CNN features and goal representation
        hidden_input = torch.cat((cnn_output, goal_layer), dim=1)
       
        # Fully Connected Layers
        h1 = self.fc1(hidden_input)
        h1 = self.dropout1(F.leaky_relu(h1, negative_slope=0.01))
        h2 = self.fc2(h1)
        h2 = self.dropout2(F.leaky_relu(h2, negative_slope=0.01)) 
        
        # Project hidden_input to match hidden_size for residual connection
        residual = self.residual_proj(hidden_input)
        lstm_input = F.leaky_relu(h2 + residual, negative_slope=0.01)
       
        # Process the concatenated input with LSTM
        if training:
           
            lstm_input = lstm_input.unsqueeze(0)   # Shape: (batch(no of agents=1), sequneclength=steps took, inputsize)
             # Initialize hidden state dynamically if not provided
            if state_in is None:
                h0 = torch.zeros(self.num_layers, lstm_input.size(0), self.hidden_size, device=input_to_nn.device)
                c0 = torch.zeros(self.num_layers, lstm_input.size(0), self.hidden_size, device=input_to_nn.device)
                state_init = (h0, c0)  # Initial hidden state
                state_in=state_init
            lstm_outputs, state_out = self.lstm(lstm_input,state_in)
            lstm_outputs = lstm_outputs.reshape(-1, self.hidden_size) # Remove the time dimension
        else:
            lstm_input = lstm_input.unsqueeze(1)  # Shape: (batch(no of agents), sequneclength=1, inputsize)
            
            if state_in is None:
                h0 = torch.zeros(self.num_layers, lstm_input.size(0), self.hidden_size, device=input_to_nn.device)
                c0 = torch.zeros(self.num_layers, lstm_input.size(0), self.hidden_size, device=input_to_nn.device)
                state_init = (h0, c0)  # Initial hidden state
                state_in=state_init
            lstm_outputs, state_out = self.lstm(lstm_input,state_in)
            lstm_outputs = lstm_outputs.reshape(-1, self.hidden_size)
        # Get the policy logits
        policy_logits = self.policy_layer(lstm_outputs)
        policy = F.softmax(policy_logits, dim=-1)
        value = self.value_layer(lstm_outputs)
        blocking = self.blocking_layer(lstm_outputs)
        on_goal = self.on_goal_layer(lstm_outputs)
        policy_sig = torch.sigmoid(self.policy_layer(lstm_outputs))
        state_in = (state_in[0].detach(), state_in[1].detach())
        state_out = (state_out[0].detach(), state_out[1].detach())
        
        
        return policy, value,state_out, blocking, on_goal,policy_sig,policy_logits


