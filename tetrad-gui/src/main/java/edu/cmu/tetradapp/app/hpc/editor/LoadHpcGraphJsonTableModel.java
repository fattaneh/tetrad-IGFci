/*
 * Copyright (C) 2015 University of Pittsburgh.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
 */
package edu.cmu.tetradapp.app.hpc.editor;

import java.util.Vector;

import javax.swing.table.DefaultTableModel;

/**
 * 
 * Feb 14, 2017 7:22:42 PM
 * 
 * @author Chirayu (Kong) Wongchokprasitti
 *
 */
public class LoadHpcGraphJsonTableModel extends DefaultTableModel {

	private static final long serialVersionUID = 2896909588298923241L;

	public LoadHpcGraphJsonTableModel(final Vector<Vector<String>> rowData, final Vector<String> columnNames) {
		super(rowData, columnNames);
	}

	public boolean isCellEditable(int row, int column) {
		return false;
	}

}
