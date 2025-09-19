# Colab Intermediate Activities

For users comfortable with Colab basics, focus on advanced Colab features, optimization techniques, and integration with external services and workflows.

## Key Learning Goals
- Advanced form widgets and interface design
- Performance optimization for compute-intensive tasks
- Integration with external data sources and APIs
- Custom visualization and interactive displays
- Workflow automation and batch processing
- Production-ready development practices
- Integration with broader research ecosystem tools

## Suggested Activity Topics

### Advanced Form Design and UI Patterns
**Concept**: Creating sophisticated interactive interfaces
**Activities**:
- Complex form layouts with conditional parameters
- Custom CSS styling for professional-looking notebooks
- Creating wizard-like workflows with multiple steps
- Building dashboard-style analysis interfaces

### Performance Optimization in Colab
**Concept**: Making the most of Colab's computational resources
**Activities**:
- GPU/TPU utilization for image processing workloads
- Memory management for large dataset processing
- Parallel processing with multiprocessing and threading
- Optimizing data loading and preprocessing pipelines

### External Data Integration
**Concept**: Connecting Colab to real-world data sources
**Activities**:
- API integration for accessing remote datasets
- Database connections (BigQuery, SQL, MongoDB)
- Cloud storage integration (AWS S3, Google Cloud Storage)
- Real-time data streaming and processing

### Custom Visualization and Reporting
**Concept**: Creating publication-ready outputs
**Activities**:
- Interactive plots with Plotly and Bokeh
- Custom HTML reports with embedded analysis
- Automated figure generation for papers
- Creating animated visualizations for presentations

### Workflow Automation and Scheduling
**Concept**: Building automated analysis pipelines
**Activities**:
- Parameterized notebooks for batch processing
- Integration with Google Sheets for data input/output
- Setting up automated email reports
- Creating reusable analysis templates

### Collaboration and Team Workflows
**Concept**: Advanced collaborative research patterns
**Activities**:
- Team notebook templates and standards
- Code review workflows in Colab
- Shared data management strategies
- Creating notebooks for different audience levels

## Activity Delivery Guidelines

**For intermediate users, emphasize:**

1. **Explain** how to scale up from simple prototypes to production workflows
2. **Demonstrate** integration patterns and best practices
3. **Challenge** with realistic multi-step projects requiring multiple Colab features

**Example structure:**
```markdown
### Activity: Building a Research Dashboard

You'll create an interactive dashboard that combines multiple data sources and provides real-time analysis capabilities.

This demonstrates how Colab can serve as a platform for sophisticated research tools.
```

```python
# Demonstration: Interactive research dashboard
#@title 📊 Research Dashboard { display-mode: "form" }
#@markdown Configure your analysis dashboard

# Advanced form parameters with conditional logic
data_source = "Local Upload" #@param ["Local Upload", "Google Drive", "URL", "BigQuery"]
analysis_mode = "Real-time" #@param ["Real-time", "Batch", "Scheduled"]
visualization_style = "Interactive" #@param ["Static", "Interactive", "Animated"]

# Conditional parameters based on data source
if data_source == "BigQuery":
    project_id = "your-project-id" #@param {type:"string"}
    dataset_name = "your-dataset" #@param {type:"string"}
elif data_source == "URL":
    data_url = "https://example.com/data.csv" #@param {type:"string"}

# Advanced visualization parameters
color_scheme = "viridis" #@param ["viridis", "plasma", "inferno", "magma"]
show_statistics = True #@param {type:"boolean"}
auto_refresh = 30 #@param {type:"slider", min:10, max:300, step:10}

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time

class ResearchDashboard:
    def __init__(self, config):
        self.config = config
        self.data_cache = {}
        self.last_update = None
    
    def load_data(self):
        """Load data based on configured source."""
        if self.config['data_source'] == "Local Upload":
            # Simulate uploaded data
            data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
                'measurement': np.random.randn(100).cumsum(),
                'category': np.random.choice(['A', 'B', 'C'], 100)
            })
        elif self.config['data_source'] == "BigQuery":
            # In real implementation, would connect to BigQuery
            data = self.simulate_bigquery_data()
        else:
            # Simulate other data sources
            data = self.simulate_generic_data()
        
        self.data_cache = data
        self.last_update = datetime.now()
        return data
    
    def create_dashboard(self):
        """Generate complete dashboard visualization."""
        data = self.load_data()
        
        # Create subplot structure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Time Series', 'Distribution', 'Category Analysis', 'Statistics'),
            specs=[[{"secondary_y": True}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # Time series plot
        fig.add_trace(
            go.Scatter(x=data['timestamp'], y=data['measurement'],
                      mode='lines+markers', name='Measurement'),
            row=1, col=1
        )
        
        # Distribution histogram
        fig.add_trace(
            go.Histogram(x=data['measurement'], name='Distribution',
                        marker_color=self.config['color_scheme']),
            row=1, col=2
        )
        
        # Category analysis
        category_stats = data.groupby('category')['measurement'].mean()
        fig.add_trace(
            go.Bar(x=category_stats.index, y=category_stats.values,
                   name='Category Means'),
            row=2, col=1
        )
        
        # Statistics table
        if self.config['show_statistics']:
            stats_data = data['measurement'].describe()
            fig.add_trace(
                go.Table(
                    header=dict(values=['Statistic', 'Value']),
                    cells=dict(values=[list(stats_data.index), 
                                     [f"{v:.3f}" for v in stats_data.values]])
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Research Dashboard - Last Updated: {self.last_update.strftime('%H:%M:%S')}",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def simulate_bigquery_data(self):
        """Simulate BigQuery data for demonstration."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='10min'),
            'measurement': np.random.exponential(2, 1000),
            'category': np.random.choice(['Treatment', 'Control'], 1000)
        })

# Create and display dashboard
dashboard_config = {
    'data_source': data_source,
    'analysis_mode': analysis_mode,
    'visualization_style': visualization_style,
    'color_scheme': color_scheme,
    'show_statistics': show_statistics,
    'auto_refresh': auto_refresh
}

dashboard = ResearchDashboard(dashboard_config)
fig = dashboard.create_dashboard()

if visualization_style == "Interactive":
    fig.show()
else:
    fig.show(renderer="png")  # Static version

print(f"✅ Dashboard created with {data_source} data source")
print(f"🔄 Mode: {analysis_mode}")
if analysis_mode == "Real-time":
    print(f"⏱️ Auto-refresh every {auto_refresh} seconds")
```

```python
# Challenge: Build a collaborative analysis workflow
#@title 🤝 Collaborative Analysis System { display-mode: "form" }
#@markdown Create a system for team-based research analysis

# TODO: Design form parameters for team collaboration
team_role = "Analyst" #@param ["Data Collector", "Analyst", "Reviewer", "Manager"]
analysis_stage = "Data Processing" #@param ["Data Collection", "Data Processing", "Analysis", "Review", "Publication"]

# TODO: Implement role-based functionality
class CollaborativeWorkflow:
    def __init__(self, role, stage):
        self.role = role
        self.stage = stage
        self.permissions = self.get_role_permissions()
    
    def get_role_permissions(self):
        # TODO: Define what each role can do at each stage
        pass
    
    def validate_data_quality(self):
        # TODO: Implement automated quality checks
        pass
    
    def create_analysis_report(self):
        # TODO: Generate role-appropriate reports
        pass
    
    def setup_review_workflow(self):
        # TODO: Create review and approval system
        pass

# TODO: Implement the collaborative features
# Consider: version control, comment systems, approval workflows
```
