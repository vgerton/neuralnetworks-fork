package com.github.samples;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import java.awt.*;

/**
 * @author kisel.nikolay@gmail.com
 * @since 06.01.2015
 */
public class Visualizer extends ApplicationFrame {

    /**
     * Creates a new plot.
     */
    public Visualizer(float[][] originalData, float[][] noisyData, float[][] resultData, boolean showLine) {

        super("Line Chart");

        final XYDataset dataset = createDataset(originalData, noisyData, resultData);
        final JFreeChart chart = createChart(dataset, showLine);
        final ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(500, 270));
        setContentPane(chartPanel);

    }

    /**
     * Creates a sample dataset.
     *
     * @return a sample dataset.
     */
    private XYDataset createDataset(float[][] originalData, float[][] noisyData, float[][] resultData) {

        final XYSeries originalSeries = new XYSeries("Original Data");
        for (int i = 0; i < originalData.length; i++) {
            originalSeries.add(originalData[i][0], originalData[i][1]);
        }

        final XYSeries noisySeries = new XYSeries("Noisy Data");
        for (int i = 0; i < noisyData.length; i++) {
            noisySeries.add(noisyData[i][0], noisyData[i][1]);
        }

        final XYSeries resultSeries = new XYSeries("Result Data");
        for (int i = 0; i < resultData.length; i++) {
            resultSeries.add(resultData[i][0], resultData[i][1]);
        }

        final XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(originalSeries);
        dataset.addSeries(noisySeries);
        dataset.addSeries(resultSeries);

        return dataset;
    }

    /**
     * Creates a chart.
     *
     * @param dataset the data for the chart.
     * @return a chart.
     */
    private JFreeChart createChart(final XYDataset dataset, boolean showLine) {

        // create the chart...
        final JFreeChart chart = ChartFactory.createXYLineChart(
                "Line Chart ",            // chart title
                "X",                      // x axis label
                "Y",                      // y axis label
                dataset,                  // data
                PlotOrientation.VERTICAL,
                true,                     // include legend
                true,                     // tooltips
                false                     // urls
        );

        // NOW DO SOME OPTIONAL CUSTOMISATION OF THE CHART...
        chart.setBackgroundPaint(Color.white);

        // get a reference to the plot for further customisation...
        final XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(Color.lightGray);
        plot.setDomainGridlinePaint(Color.white);
        plot.setRangeGridlinePaint(Color.white);

        final XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        if (!showLine) {
            renderer.setSeriesLinesVisible(0, false);
            renderer.setSeriesLinesVisible(1, false);
            renderer.setSeriesLinesVisible(2, false);
        }
//        renderer.setSeriesShapesVisible(1, false);
        plot.setRenderer(renderer);


        return chart;
    }

    /**
     * Create Chart.
     */
    public static void createChart(float[][] originalData, float[][] noisyData, float[][] resultData, boolean showLine) {
        final Visualizer demo = new Visualizer(originalData, noisyData, resultData, showLine);
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);
    }

}
