def get_metrics_summary(self):
        """
        Get summary of monitoring metrics
        
        Returns:
            Dictionary with metrics summary
        """
        if len(self.metrics_history['errors']) < 2:
            return {
                'count': 0,
                'rmse': None,
                'mae': None,
                'bias': None,
                'alert_count': 0
            }
            
        return {
            'count': len(self.metrics_history['errors']),
            'rmse': self.metrics_history['rmse'][-1] if self.metrics_history['rmse'] else None,
            'mae': self.metrics_history['mae'][-1] if self.metrics_history['mae'] else None,
            'bias': self.metrics_history['bias'][-1] if self.metrics_history['bias'] else None,
            'alert_count': len(self.alerts)
        }
        
    def plot_metrics(self, save_path=None):
        """
        Plot monitoring metrics
        
        Args:
            save_path: Path to save plot (if None, display plot)
            
        Returns:
            Plot figure
        """
        if len(self.metrics_history['errors']) < 10:
            logging.warning("Not enough data to plot metrics")
            return None
            
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Get timestamps
        timestamps = self.metrics_history['timestamps']
        
        # Plot 1: Predictions vs Targets
        axes[0].plot(timestamps, self.metrics_history['predictions'], label='Predictions', alpha=0.7)
        axes[0].plot(timestamps, self.metrics_history['targets'], label='Targets', alpha=0.7)
        axes[0].set_title('Predictions vs Targets')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot 2: Errors
        axes[1].plot(timestamps, self.metrics_history['errors'], color='red', alpha=0.7)
        axes[1].set_title('Prediction Errors')
        axes[1].set_ylabel('Error')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].grid(True)
        
        # Plot 3: RMSE and Bias
        ax3 = axes[2]
        ax3.plot(timestamps[1:], self.metrics_history['rmse'], label='RMSE', color='purple', alpha=0.7)
        ax3.set_ylabel('RMSE', color='purple')
        ax3.tick_params(axis='y', labelcolor='purple')
        ax3.set_title('RMSE and Bias')
        ax3.grid(True)
        
        # Create secondary y-axis for bias
        ax3b = ax3.twinx()
        ax3b.plot(timestamps[1:], self.metrics_history['bias'], label='Bias', color='green', alpha=0.7)
        ax3b.set_ylabel('Bias', color='green')
        ax3b.tick_params(axis='y', labelcolor='green')
        
        # Add alert markers
        for alert in self.alerts:
            alert_time = alert['timestamp']
            severity_color = 'red' if alert['severity'] == 'high' else 'orange'
            
            # Find index of alert timestamp
            try:
                idx = timestamps.index(alert_time)
                
                # Add marker to error plot
                axes[1].scatter(
                    alert_time, 
                    self.metrics_history['errors'][idx], 
                    color=severity_color, 
                    s=100, 
                    marker='o', 
                    zorder=5
                )
            except ValueError:
                pass
                
        # Set common x-axis label
        fig.text(0.5, 0.04, 'Time', ha='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
        return fig
        
    def load(self, filepath):
        """Load the saved state of the model monitor"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            # Update attributes
            self.window_size = state.get('window_size', self.window_size)
            self.alert_threshold = state.get('alert_threshold', self.alert_threshold)
            self.metrics_history = state.get('metrics_history', self.metrics_history)
            self.alerts = state.get('alerts', self.alerts)
            self.market_conditions = state.get('market_conditions', [])
            
            logging.info(f"Model monitor state loaded from {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading model monitor state: {e}")
            return False
