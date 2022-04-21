///////////////////////////////////////////////////////////////////////////////
// For information as to what this class does, see the Javadoc, below.       //
// Copyright (C) 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,       //
// 2007, 2008, 2009, 2010, 2014, 2015 by Peter Spirtes, Richard Scheines, Joseph   //
// Ramsey, and Clark Glymour.                                                //
//                                                                           //
// This program is free software; you can redistribute it and/or modify      //
// it under the terms of the GNU General Public License as published by      //
// the Free Software Foundation; either version 2 of the License, or         //
// (at your option) any later version.                                       //
//                                                                           //
// This program is distributed in the hope that it will be useful,           //
// but WITHOUT ANY WARRANTY; without even the implied warranty of            //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             //
// GNU General Public License for more details.                              //
//                                                                           //
// You should have received a copy of the GNU General Public License         //
// along with this program; if not, write to the Free Software               //
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA //
///////////////////////////////////////////////////////////////////////////////
package edu.cmu.tetradapp.editor;

import edu.cmu.tetrad.bayes.BayesIm;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetradapp.workbench.GraphWorkbench;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.border.MatteBorder;
import javax.swing.table.TableCellEditor;

/**
 * Allows the user to choose a variable in a Bayes net and edit the parameters
 * associated with that variable. Parameters are of the form
 * P(Node=value1|Parent1=value2, Parent2=value2,...); values for these
 * parameters are probabilities ranging from 0.0 to 1.0. For a given combination
 * of parent values for node N, the probabilities for the values of N
 * conditional on that combination of parent values must sum to 1.0
 *
 * @author Joseph Ramsey jdramsey@andrew.cmu.edu
 */
final class BayesEstimatorEditorWizard extends JPanel {

    private static final long serialVersionUID = 5206106976460658161L;

    private BayesIm bayesIm;
    private JComboBox<Node> varNamesComboBox;
    private GraphWorkbench workbench;
    private BayesImNodeEditingTable editingTable;
    private JPanel tablePanel;

    private boolean enableEditing = true;

    public BayesEstimatorEditorWizard(BayesIm bayesIm, GraphWorkbench workbench) {
        if (bayesIm == null) {
            throw new NullPointerException();
        }

        if (workbench == null) {
            throw new NullPointerException();
        }

        workbench.setAllowDoubleClickActions(false);
        setBorder(new MatteBorder(10, 10, 10, 10, getBackground()));
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setFont(new Font("SanSerif", Font.BOLD, 12));

        // Set up components.
        this.varNamesComboBox = createVarNamesComboBox(bayesIm);
        workbench.scrollWorkbenchToNode((Node) varNamesComboBox.getSelectedItem());

        JButton nextButton = new JButton("Next");
        nextButton.setMnemonic('N');

        Node node = (Node) varNamesComboBox.getSelectedItem();
        editingTable = new BayesImNodeEditingTable(node, bayesIm);
        editingTable.addPropertyChangeListener((evt) -> {
            if ("modelChanged".equals(evt.getPropertyName())) {
                firePropertyChange("modelChanged", null, null);
            }
        });

        JScrollPane scroll = new JScrollPane(editingTable);
        scroll.setPreferredSize(new Dimension(0, 150));
        tablePanel = new JPanel();
        tablePanel.setLayout(new BorderLayout());
        tablePanel.add(scroll, BorderLayout.CENTER);
        editingTable.grabFocus();

        // Do Layout.
        Box b1 = Box.createHorizontalBox();
        b1.add(new JLabel("Choose the next variable to view:  "));
        b1.add(varNamesComboBox);
        b1.add(nextButton);
        b1.add(Box.createHorizontalGlue());

        Box b4 = Box.createHorizontalBox();
        b4.add(tablePanel, BorderLayout.CENTER);

        add(b1);
        add(Box.createVerticalStrut(10));
        add(b4);

        // Add listeners.
        varNamesComboBox.addActionListener((e) -> {
            getWorkbench().scrollWorkbenchToNode((Node) varNamesComboBox.getSelectedItem());
            setCurrentNode((Node) varNamesComboBox.getSelectedItem());
        });

        nextButton.addActionListener((e) -> {
            int current = varNamesComboBox.getSelectedIndex();
            int max = varNamesComboBox.getItemCount();

            ++current;

            if (current == max) {
                JOptionPane.showMessageDialog(BayesEstimatorEditorWizard.this,
                        "There are no more variables.");
            }

            int set = (current < max) ? current : 0;

            varNamesComboBox.setSelectedIndex(set);
        });

        workbench.addPropertyChangeListener((evt) -> {
            if (evt.getPropertyName().equals("selectedNodes")) {
                List selection = (List) (evt.getNewValue());
                if (selection.size() == 1) {
                    varNamesComboBox.setSelectedItem((Node) selection.get(0));
                }
            }
        });

        this.bayesIm = bayesIm;
        this.workbench = workbench;
    }

    private JComboBox<Node> createVarNamesComboBox(BayesIm bayesIm) {
        JComboBox<Node> varNameComboBox = new JComboBox<>();
        varNameComboBox.setBackground(Color.white);

        Graph graph = bayesIm.getBayesPm().getDag();

        List<Node> nodes = graph.getNodes().stream().collect(Collectors.toList());
        Collections.sort(nodes);
        nodes.forEach(varNameComboBox::addItem);

        if (varNameComboBox.getItemCount() > 0) {
            varNameComboBox.setSelectedIndex(0);
        }

        return varNameComboBox;
    }

    /**
     * Sets the getModel display to reflect the stored values of the getModel
     * node.
     */
    private void setCurrentNode(Node node) {
        TableCellEditor cellEditor = editingTable.getCellEditor();

        if (cellEditor != null) {
            cellEditor.cancelCellEditing();
        }

        editingTable = new BayesImNodeEditingTable(node, getBayesIm());
        editingTable.addPropertyChangeListener((evt) -> {
            if ("modelChanged".equals(evt.getPropertyName())) {
                firePropertyChange("modelChanged", null, null);
            }
        });

        JScrollPane scroll = new JScrollPane(editingTable);
        scroll.setPreferredSize(new Dimension(0, 150));

        tablePanel.removeAll();
        tablePanel.add(scroll, BorderLayout.CENTER);
        tablePanel.revalidate();
        tablePanel.repaint();

        editingTable.grabFocus();
    }

    private BayesIm getBayesIm() {
        return bayesIm;
    }

    private GraphWorkbench getWorkbench() {
        return workbench;
    }

    public boolean isEnableEditing() {
        return enableEditing;
    }

    public void enableEditing(boolean enableEditing) {
        this.enableEditing = enableEditing;
        if (this.workbench != null) {
            this.workbench.enableEditing(enableEditing);
        }
    }

}
