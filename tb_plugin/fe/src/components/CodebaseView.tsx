/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import Card from '@material-ui/core/Card'
import CardContent from '@material-ui/core/CardContent'
import CardHeader from '@material-ui/core/CardHeader'
import Grid from '@material-ui/core/Grid'
import { makeStyles } from '@material-ui/core/styles'
import CardMedia from '@material-ui/core/CardMedia'
import * as React from 'react'
import * as api from '../api'
import { AntTableChart } from './charts/AntTableChart'
import { DataLoading } from './DataLoading'
import { TextListItem } from './TextListItem'
import { Table } from 'antd'
import TextField, {
  StandardTextFieldProps,
  TextFieldProps
} from '@material-ui/core/TextField'
import { useSearch } from '../utils/search'

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1
  },
  inputWidthOverflow: {
    minWidth: '15em',
    whiteSpace: 'nowrap'
  },
  hide: {
    display: 'none'
  }
}))

export interface IProps {
  run: string
  worker: string
  span: string
}

const getKeyedTableColumns = (columns: api.KeyedColumn[]) => {
  return columns.map((col) => {
    return {
      dataIndex: col.key,
      key: col.key,
      title: col.name
    }
  })
}

const getTableRows = (rows: api.PStatsTree[]) => {
  return rows.map((row) => {
    const data: any = {
      key: row.key,
      filepath: row.filepath,
      func_name: row.func_name,
      nc: row.nc,
      cc: row.cc,
      tt: row.tt,
      ct: row.ct,
      time_per_call: row.time_per_call,
      ct_ratio: row.ct_ratio,
      time_per_prim_call: row.time_per_prim_call
    }

    if (row.children.length) {
      data.children = getTableRows(row.children)
    }

    return data
  })
}
export const CodebaseView: React.FC<IProps> = (props) => {
  const { run, worker, span } = props
  const [pythonBottleneck, setPythonBottleneck] = React.useState<
    api.PythonBottleneck | undefined
  >(undefined)
  const [pStatsGraph, setPStatsGraph] = React.useState<api.Graph | undefined>(
    undefined
  )
  const [sortColumn, setSortColumn] = React.useState('')
  const [pStatsOverViews, setPStatsOverViews] = React.useState<
    api.PStatsOverview[]
  >([])
  const [rows, setRows] = React.useState<any[]>([])
  const [columns, setColumns] = React.useState<any[]>([])
  const [searchFuncName, setSearchFuncName] = React.useState('')

  const [searchedFuncTable] = useSearch(searchFuncName, 'Function', pStatsGraph)

  const onSearchFuncChanged: TextFieldProps['onChange'] = (event) => {
    setSearchFuncName(event.target.value as string)
  }

  React.useEffect(() => {
    api.defaultApi.codebaseGet(run, worker, span).then((resp) => {
      setPythonBottleneck(resp.python_bottleneck)
    })
  }, [run, worker, span])
  const classes = useStyles()

  React.useEffect(() => {
    if (pythonBottleneck) {
      setPStatsGraph(pythonBottleneck.pstats.data)
      setSortColumn(pythonBottleneck.pstats.metadata.sort)
      setPStatsOverViews(pythonBottleneck.pstats.overview)
      setColumns(getKeyedTableColumns(pythonBottleneck.pstats.columns))
      setRows(getTableRows(pythonBottleneck.pstats.tree))
    }
  }, [pythonBottleneck])

  return (
    <div className={classes.root}>
      <Card variant="outlined">
        <CardHeader title="Python Bottleneck Analaysis" />
        <CardContent>
          <Grid container spacing={1}>
            <Grid container item>
              {pythonBottleneck && (
                <Grid item sm={12}>
                  <CardMedia
                    component="img"
                    src={`data:image/png;base64, ${pythonBottleneck.image_content}`}
                    alt="python profiling graph"
                  />
                </Grid>
              )}
            </Grid>
          </Grid>
          <Grid container item spacing={1}>
            <Grid item sm={12}>
              {React.useMemo(
                () => (
                  <Card variant="outlined">
                    <CardHeader title="Summary" />
                    <CardContent>
                      {pStatsOverViews.map((item) => (
                        <TextListItem name={item.title} value={item.value} />
                      ))}
                    </CardContent>
                  </Card>
                ),
                [pStatsOverViews]
              )}
            </Grid>
          </Grid>
          <Grid item sm={12}>
            {rows && rows.length > 0 && (
              <Table
                size="small"
                bordered
                columns={columns}
                dataSource={rows}
                expandable={{
                  defaultExpandAllRows: false
                }}
              />
            )}
          </Grid>
          <Grid item container direction="column" spacing={1}>
            <Grid container justify="space-around">
              <Grid item>
                <TextField
                  classes={{ root: classes.inputWidthOverflow }}
                  value={searchFuncName}
                  onChange={onSearchFuncChanged}
                  type="search"
                  label="Search by Function"
                />
              </Grid>
            </Grid>
          </Grid>
          <Grid item container direction="column" spacing={1} sm={12}>
            <Grid item>
              <DataLoading value={searchedFuncTable}>
                {(graph) => (
                  <AntTableChart graph={graph} sortColumn={sortColumn} />
                )}
              </DataLoading>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </div>
  )
}
