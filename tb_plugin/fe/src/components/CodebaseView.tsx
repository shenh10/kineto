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

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1
  }
}))

export interface IProps {
  run: string
  worker: string
  span: string
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
          <Grid item container direction="column" spacing={1} sm={12}>
            <Grid item>
              <DataLoading value={pStatsGraph}>
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
