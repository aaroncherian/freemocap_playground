from pathlib import Path
import pandas as pd 

path_to_diagnostics_folder =  Path(__file__).parent/'diagnostic_results'

results_list = list(path_to_diagnostics_folder.glob('*.csv'))
total_df = pd.DataFrame()
for result in results_list:
    result_df = pd.read_csv(result)
    total_df = pd.concat([total_df, result_df], ignore_index=True)

 
f = 2
import plotly.express as px
def plot_jerk_trends(total_df):
    # Reset index for easy handling in Plotly
    total_df = total_df[total_df['name']=='total']
    # Line plot of jerk trends per joint across versions
    fig = px.line(total_df, x="version", y="mean_jerk", color="data_stage",
                  title="Jerk Trends Across Versions", markers=True)

    return fig



def format_jerk_table(total_df):
    """
    Formats the pivoted DataFrame into a nicely styled HTML table with alternating column colors for 'raw' and 'processed'.
    """
    # Pivot the data for a clean summary table
    table_df = total_df.pivot(index="name", columns=["version", "data_stage"], values="mean_jerk")

    # Identify 'raw' and 'processed' columns
    raw_columns = [col for col in table_df.columns if col[1] == "raw"]
    processed_columns = [col for col in table_df.columns if col[1] == "processed"]

    # Define alternating column colors
    def highlight_columns(s):
        color_raw = 'background-color: #D6EAF8'  # Light blue
        color_processed = 'background-color: #FAD7A0'  # Light orange
        return [color_raw if col in raw_columns else color_processed if col in processed_columns else '' for col in s.index]

    def bold_total(s):
        return ['font-weight: bold' if s.name == "total" else '' for _ in s]

    # Style the DataFrame
    styled_table = (
        table_df.style
        .set_table_styles([
            {"selector": "thead th", "props": [("font-weight", "bold"), ("background-color", "#f4f4f4"), ("text-align", "center")]},
            {"selector": "tbody td", "props": [("text-align", "center"), ("min-width", "80px"), ("max-width", "120px")]},
            {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f9f9f9")]}
        ])
        .apply(highlight_columns, axis=1)
        .apply(bold_total, axis = 1)
        .format("{:.2f}")  # Round values to 2 decimal places
    )

    # Convert styled DataFrame to HTML
    return styled_table.to_html()



from jinja2 import Template

def generate_html_report(total_df, output_path="diagnostic_report.html"):
    jerk_plot = plot_jerk_trends(total_df).to_html(full_html=False)
    html_table = format_jerk_table(total_df)

    html_template = """
    <html>
    <head>
        <title>Motion Capture Diagnostic Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            .plot { margin-bottom: 40px; }
            .table-container { margin-top: 40px; }
        </style>
    </head>
    <body>
        <h1>Motion Capture Diagnostic Report</h1>
        
        <h2>Jerk Trends Across Versions</h2>
        <div class="plot">{{ jerk_plot|safe }}</div>
        
        <h2>Summary Table: Jerk Across Versions</h2>
        <div class="table-container">{{ html_table|safe }}</div>
    </body>
    </html>
    """

    # Render the HTML
    template = Template(html_template)
    rendered_html = template.render(jerk_plot=jerk_plot, html_table=html_table)

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    print(f"Report saved at: {output_path}")

# Generate the report
generate_html_report(total_df)